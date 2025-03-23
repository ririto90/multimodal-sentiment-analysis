# instructor.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import DataSet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

from transformers import BertModel
from torchvision import models

# Added for class weighting and LR scheduling
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_utils import DataSet, MVSAAdapter
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

def weighted_macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, 
        preds, 
        average='weighted',
        zero_division=0
    )
    return f_macro

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.max_val_f1 = 0
        self.max_test_f1 = 0
        self.writer = SummaryWriter(log_dir=opt.log_dir)

        print('> training arguments:')
        for arg in vars(opt):
            print(f'>>> {arg}: {getattr(opt, arg)}')

        # 1) Build your custom DataSet
        my_data = DataSet(
            dataset_name=opt.dataset
        )
        opt.num_classes = 3  # or however many classes you have

        # 2) Create Dataset objects for train/val/test with your adapter
        train_set = MVSAAdapter(
            my_data.text_train,
            my_data.images_train,
            my_data.post_train_labels,
            opt.max_seq_len
        )
        val_set = MVSAAdapter(
            my_data.text_validation,
            my_data.images_validation,
            my_data.post_val_labels,
            opt.max_seq_len
        )
        test_set = MVSAAdapter(
            my_data.text_test,
            my_data.images_test,
            my_data.post_test_labels,
            opt.max_seq_len
        )

        # 3) Wrap them in DataLoaders
        self.train_data_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
        self.val_data_loader   = DataLoader(val_set,   batch_size=opt.batch_size, shuffle=False)
        self.test_data_loader  = DataLoader(test_set,  batch_size=opt.batch_size, shuffle=False)

        print('Building model')

        # Text encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Freeze BERT if specified
        if hasattr(opt, 'freeze_bert') and opt.freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Weight matrix for attention mechanism
        self.W = nn.Linear(768, 768) # Update every iteration
        
        # Image encoder
        self.resnet = models.resnet152(pretrained=True)
        # Replace final FC layer with identity, so we get a 2048-dim feature
        self.resnet.fc = nn.Identity()
        # Drop the last two layers (avgpool + fc) so you get a 7×7 spatial grid
        modules = list(self.resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Model head
        self.model = opt.model_class(opt)
        
        self.opt.counter += 1
        if self.opt.counter < 3:
            print(self.opt.counter)

        # Use multiple GPUs if available
        if torch.cuda.device_count() > 1:
            self.bert = nn.DataParallel(self.bert)
            self.W = nn.DataParallel(self.W)
            self.resnet = nn.DataParallel(self.resnet)
            self.model = nn.DataParallel(self.model)

        self.bert.to(device)
        self.W.to(device)
        self.resnet.to(device)
        self.model.to(device)
        
        # Set up optimizer
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = opt.optimizer(params, lr=opt.learning_rate)

        # Set up a learning rate scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1)

        # We keep a criterion for evaluation (no class weighting in eval)
        self.criterion = nn.CrossEntropyLoss()

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

                # Bert
                input_ids = sample_batched['input_ids'].to(device)
                attention_mask = sample_batched['attention_mask'].to(device)
                bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
                text_features = bert_outputs.last_hidden_state  # shape [B, seq_len, 768]
                
                # Resnet
                images = sample_batched['images'].to(device)  # [B, 3, H, W]
                image_feature = self.resnet(images)  # shape [B, 2048, 7, 7]
                B, C, H, W = image_feature.shape     # [B, 2048, 7, 7]
                image_features = image_feature.view(B, H*W, C)  # reshape -> [B, 49, 2048]
                
                outputs = self.model(text_features, image_features)

                ### Backprop with class weighting ###
                targets = sample_batched["polarity"].to(device).long()
                labels_cpu = targets.cpu().numpy()

                # Compute class weights for balanced training
                unique_labels = np.unique(labels_cpu)
                class_weights_batch = compute_class_weight(
                    class_weight='balanced',
                    classes=unique_labels,     # e.g. [0,1], or [1,2], etc.
                    y=labels_cpu
                )
                full_weights = torch.ones(self.opt.num_classes, dtype=torch.float)
                
                for idx, label_val in enumerate(unique_labels):
                    full_weights[label_val] = class_weights_batch[idx]
                full_weights = full_weights.to(device)

                loss = F.cross_entropy(outputs, targets, weight=full_weights)
                loss.backward()
                self.optimizer.step()

                global_step += 1
                self.writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                
                # Accumulate for epoch-average
                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size

                batch_end_time = time.time()

                # Evaluate at intervals
                if i_batch % self.opt.log_step == 0:
                    val_acc, val_f1_macro, val_f1_weighted, val_loss = self.evaluate(self.val_data_loader)
                    test_acc, test_f1, test_loss = self.evaluate(self.test_data_loader)

                    self.writer.add_scalar('Loss/val_log_step', val_loss, global_step)
                    print(f'Batch {i_batch} completed in {batch_end_time - batch_start_time:.2f} seconds '
                          f'({(batch_end_time - batch_start_time) / 60:.2f} minutes)')

                    if val_f1 > self.max_val_f1:
                        print(f"Val Acc: {val_acc:.4f}, Macro-F1: {val_f1_macro:.4f}, Weighted-F1: {val_f1_weighted:.4f}, Loss: {val_loss:.4f}")
                        self.max_val_f1 = val_f1_macro
                        self.max_val_weighted_f1 = val_f1_weighted
                        self.max_test_f1 = test_f1

                    print(f'loss: {loss.item():.6f}, val_acc: {val_acc * 100:.2f}% ({val_acc:.6f}), '
                          f'Val Acc: {val_acc:.4f}, Macro-F1: {val_f1_macro:.4f}, Weighted-F1: {val_f1_weighted:.4f}, Loss: {val_loss:.4f}'
                          f'({test_acc:.6f}), test_f1: {test_f1 * 100:.2f}% ({test_f1:.6f})')
            
            # End of epoch compute epoch-average training loss
            epoch_train_loss = running_loss / total_samples

            # Evaluate on the validation set
            val_acc, val_f1_macro, val_f1_weighted, val_loss = self.evaluate(self.val_data_loader)

            # **Epoch-level** logging to TensorBoard
            self.writer.add_scalar('Loss/train_epoch', epoch_train_loss, epoch)
            self.writer.add_scalar('Loss/val_epoch', val_loss, epoch)

            # Step the scheduler with validation loss
            self.scheduler.step(val_loss)

            epoch_end_time = time.time()
            print(f'Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds '
                  f'({(epoch_end_time - epoch_start_time) / 60:.2f} minutes)')

        self.writer.flush()

        print(f"RESULT: Max Val F1: {self.max_val_f1:.6f}, Max Weighted F1: {self.max_val_weighted_f1:.6f}, Max Test F1: {self.max_test_f1:.6f}")
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
                text_features = bert_outputs.last_hidden_state
                
                # Resnet
                images = sample_batched['images'].to(device)
                image_feature = self.resnet(images)
                B, C, H, W = image_feature.shape
                image_features = image_feature.view(B, H*W, C)

                outputs = self.model(text_features, image_features)

                targets = sample_batched["polarity"].to(device).long()
                # For validation/test, we do not use class weights
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)

                true_labels.append(targets.cpu().numpy())
                pred_probs.append(outputs.cpu().numpy())

        true_labels = np.concatenate(true_labels)
        pred_probs = np.concatenate(pred_probs)
        
        f1_score_macro = macro_f1(true_labels, pred_probs)
        f1_score_weighted = weighted_macro_f1(true_labels, pred_probs)
        accuracy = (np.argmax(pred_probs, axis=-1) == true_labels).mean()
        avg_loss = total_loss / len(data_loader.dataset)
        
        if return_labels:
            return accuracy, f1_score_macro, avg_loss, true_labels, pred_probs
        else:
            # Return 4 things: accuracy, macro-F1, weighted-F1, and loss
            return accuracy, f1_score_macro, f1_score_weighted, avg_loss

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