# instructor.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

from transformers import BertModel
from torchvision import models

from data_utils import MVSADatasetReader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from Project.settings import DATASET_PATHS, IMAGE_PATH

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Number of GPUs available: {torch.cuda.device_count()}')

def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, 
        preds, 
        average='macro',
        zero_division=0
    )
    return f_macro

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.max_val_f1 = 0
        self.max_test_f1 = 0

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=opt.log_dir)

        print('> training arguments:')
        for arg in vars(opt):
            print(f'>>> {arg}: {getattr(opt, arg)}')

        # Load dataset
        mvsa_dataset = MVSADatasetReader(paths=DATASET_PATHS, path_image=IMAGE_PATH, dataset=opt.dataset, max_seq_len=opt.max_seq_len)
        opt.num_classes = mvsa_dataset.num_classes
        self.class_weights = mvsa_dataset.class_weights.to(device).float()

        self.train_data_loader = DataLoader(dataset=mvsa_dataset.train_data, 
                                            batch_size=opt.batch_size, shuffle=True)
        self.val_data_loader = DataLoader(dataset=mvsa_dataset.val_data, 
                                          batch_size=opt.batch_size, shuffle=False)
        self.test_data_loader = DataLoader(dataset=mvsa_dataset.test_data, 
                                           batch_size=opt.batch_size, shuffle=False)

        print('Building model')

        # Text encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Image encoder
        self.resnet = models.resnet50(pretrained=True)
        
        # Replace final FC layer with identity, so we get a 2048-dim feature
        self.resnet.fc = nn.Identity()

        # Model head (unchanged, but you must ensure `opt.model_class` can
        # accept the concatenated [BERT + ResNet] feature dimensions)
        self.model = opt.model_class(opt)
        
        self.opt.counter += 1
        if self.opt.counter < 3:
            print(self.opt.counter)


        # Use multiple GPUs if available
        if torch.cuda.device_count() > 1:
            self.bert = nn.DataParallel(self.bert)
            self.resnet = nn.DataParallel(self.resnet)
            self.model = nn.DataParallel(self.model)

        self.bert.to(device)
        self.resnet.to(device)
        self.model.to(device)
        
        # Loss & optimizer
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = opt.optimizer(params, lr=opt.learning_rate)

        self.reset_parameters()

    def reset_parameters(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
                if len(p.shape) > 1:
                    self.opt.initializer(p)
            else:
                n_nontrainable_params += n_params
        print(f'n_trainable_params: {n_trainable_params}, n_nontrainable_params: {n_nontrainable_params}')

    def run(self):
        global_step = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print(f'epoch: {epoch}')
            epoch_start_time = time.time()

            self.model.train()
            running_loss = 0.0
            total_samples = 0
            
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                batch_start_time = time.time()
                
                self.optimizer.zero_grad()
                
                # batch_labels = sample_batched['polarity'].cpu().numpy()
                # unique_lbls, counts = np.unique(batch_labels, return_counts=True)
                # print(f"[DEBUG] Batch {i_batch} distribution = {dict(zip(unique_lbls, counts))}")

                # Bert
                input_ids = sample_batched['input_ids'].to(device)
                attention_mask = sample_batched['attention_mask'].to(device)
                bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
                text_feature = bert_outputs.pooler_output  # [B, 768]
                
                # Resnet
                images = sample_batched['images'].to(device)  # [B, 3, H, W]
                image_feature = self.resnet(images)  # [B, 2048]
                
                outputs = self.model(text_feature, image_feature)
                
                pred = outputs.argmax(dim=-1)
                if i_batch == 0 and (epoch == 0 or epoch == self.opt.num_epoch-1):
                    print("[DEBUG] Sample predictions in evaluate: ", pred[:10])
                
                # Debug:
                if i_batch == 0 and epoch == 0:
                    print(f"[DEBUG] outputs.shape: {outputs.shape}") 
                    print("[DEBUG] Sample of raw logits (first 5):")
                    print(outputs[:5])
                    
                if i_batch == 0 and epoch == 0:
                    probs = F.softmax(outputs, dim=-1)
                    print("[DEBUG] Sample of predicted probabilities (first 5):")
                    print(probs[:5])
                
                ### Backprop ###
                targets = sample_batched["polarity"].to(device).long()
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         if torch.isnan(param.grad).any():
                #             print(f"[DEBUG][WARNING] {name} gradient has NaNs!")
                #         if torch.all(param.grad == 0):
                #             print(f"[DEBUG][WARNING] {name} gradient is all zero!")
                #     else:
                #         print(f"[DEBUG] {name} has no grad (param.grad is None)")
                
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.clip_grad)
                self.optimizer.step()

                global_step += 1
                self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                
                # Accumulate for epoch-average
                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size

                batch_end_time = time.time()
            

                if i_batch % self.opt.log_step == 0:
                    val_acc, val_f1, val_loss = self.evaluate(self.val_data_loader)
                    test_acc, test_f1, test_loss = self.evaluate(self.test_data_loader)

                    self.writer.add_scalar('Loss/val_log_step', val_loss, global_step)

                    print(f'Batch {i_batch} completed in {batch_end_time - batch_start_time:.2f} seconds '
                          f'({(batch_end_time - batch_start_time) / 60:.2f} minutes)')

                    if val_f1 > self.max_val_f1:
                        print(f"New best val_f1: {val_f1:.6f} (previous best: {self.max_val_f1:.6f})")
                        self.max_val_f1 = val_f1
                        self.max_test_f1 = test_f1

                    print(f'loss: {loss.item():.6f}, val_acc: {val_acc * 100:.2f}% ({val_acc:.6f}), '
                          f'val_f1: {val_f1 * 100:.2f}% ({val_f1:.6f}), test_acc: {test_acc * 100:.2f}% '
                          f'({test_acc:.6f}), test_f1: {test_f1 * 100:.2f}% ({test_f1:.6f})')
            
            # ---------- [End of epoch] ----------    
            # End of epoch compute epoch-average training loss
            epoch_train_loss = running_loss / total_samples

            # Evaluate on the validation set
            val_acc, val_f1, val_loss = self.evaluate(self.val_data_loader)
            # test_acc, test_f1, test_loss = self.evaluate(self.test_data_loader)

            # **Epoch-level** logging to TensorBoard
            self.writer.add_scalar('Loss/train_epoch', epoch_train_loss, epoch)
            self.writer.add_scalar('Loss/val_epoch', val_loss, epoch)

            epoch_end_time = time.time()
            print(f'Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds '
                  f'({(epoch_end_time - epoch_start_time) / 60:.2f} minutes)')

        self.writer.flush()

        print(f"RESULT: Max Val F1: {self.max_val_f1:.6f}, Max Test F1: {self.max_test_f1:.6f}")
        print("Training complete. Generating confusion matrix on the test set.")

        test_acc, test_f1, test_loss, true_labels, pred_probs = self.evaluate(self.test_data_loader, return_labels=True)
        pred_labels = np.argmax(pred_probs, axis=-1)

        classes = [str(i) for i in range(self.opt.num_classes)]

        self.plot_confusion_matrix(true_labels, pred_labels, classes, self.opt.log_dir)

    def evaluate(self, data_loader, return_labels=False):
        self.model.eval()
        true_labels, pred_probs = [], []
        total_loss = 0.0
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_loader):
                
                # Bert
                input_ids = sample_batched['input_ids'].to(device)
                attention_mask = sample_batched['attention_mask'].to(device)
                bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
                text_feature = bert_outputs.pooler_output  # [B, 768]
                
                # Resnet
                images = sample_batched['images'].to(device)  # [B, 3, H, W]
                image_feature = self.resnet(images)  # [B, 2048]
                
                outputs = self.model(text_feature, image_feature)

                ### Backprop ###
                targets = sample_batched["polarity"].to(device).long()
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * targets.size(0)

                true_labels.append(targets.cpu().numpy())
                pred_probs.append(outputs.cpu().numpy())

        true_labels = np.concatenate(true_labels)
        pred_probs = np.concatenate(pred_probs)

        f1_score = macro_f1(true_labels, pred_probs)
        accuracy = (np.argmax(pred_probs, axis=-1) == true_labels).mean()
        avg_loss = total_loss / len(data_loader.dataset)

        if return_labels:
            return accuracy, f1_score, avg_loss, true_labels, pred_probs
        else:
            return accuracy, f1_score, avg_loss

    def read_tensorboard_loss(self, log_dir=None):
        print('Reading TensorBoard loss at each epoch:')
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        print('Available tags:', ea.Tags())

        loss_events = ea.Scalars('Loss/train_epoch')

    def plot_tensorboard_loss(self, log_dir=None):
        output_file = os.path.join(log_dir, 'trainval_loss_curves.png')
        print(f"Output File: {output_file}")

        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        train_loss_events = ea.Scalars('Loss/train_epoch')
        val_loss_events = ea.Scalars('Loss/val_epoch')

        train_epochs = [event.step for event in train_loss_events]
        train_losses = [event.value for event in train_loss_events]
        val_epochs = [event.step for event in val_loss_events]
        val_losses = [event.value for event in val_loss_events]

        plt.figure(figsize=(12, 6))
        plt.plot(train_epochs, train_losses, label='Training Loss', alpha=0.7)
        plt.plot(val_epochs, val_losses, label='Validation Loss', color='orange', alpha=0.7)

        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        ax.grid(True, which='major', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file)
        print(f'Training and validation loss curves saved to {output_file}')

    def plot_confusion_matrix(self, true_labels, pred_labels, classes, log_dir):
        
        cm = confusion_matrix(true_labels, pred_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Normalized Confusion Matrix')

        output_file = os.path.join(log_dir, 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f'Confusion matrix saved to {output_file}')
