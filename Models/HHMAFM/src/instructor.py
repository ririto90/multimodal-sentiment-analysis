# instructor.py

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torchvision.models import ResNet152_Weights, DenseNet121_Weights
from transformers import RobertaModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

from data_utils import MVSADatasetReader
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Number of GPUs available: {torch.cuda.device_count()}')

def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, preds, average='macro')
    return f_macro

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.max_dev_f1 = 0
        self.max_test_f1 = 0

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=opt.log_dir)

        print('> training arguments:')
        for arg in vars(opt):
            print(f'>>> {arg}: {getattr(opt, arg)}')

        # Data transformations
        transform = transforms.Compose([
            transforms.RandomCrop(opt.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229, 0.224, 0.225))
        ])

        # Load dataset
        mvsa_dataset = MVSADatasetReader(transform, dataset=opt.dataset, 
                                         max_seq_len=opt.max_seq_len, path_image=opt.path_image)
        opt.num_classes = mvsa_dataset.num_classes

        self.train_data_loader = DataLoader(dataset=mvsa_dataset.train_data, 
                                            batch_size=opt.batch_size, shuffle=True)
        self.dev_data_loader = DataLoader(dataset=mvsa_dataset.dev_data, 
                                          batch_size=opt.batch_size, shuffle=False)
        self.test_data_loader = DataLoader(dataset=mvsa_dataset.test_data, 
                                           batch_size=opt.batch_size, shuffle=False)

        print('Building model')
        
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        text_dim = self.roberta.config.hidden_size
        
        self.resnet = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        resnet_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.densenet = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        densenet_dim = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()
        
        self.model = opt.model_class(opt, text_dim, resnet_dim, densenet_dim)
        
        # Use multiple GPUs if available
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            self.roberta = nn.DataParallel(self.roberta)
            self.resnet = nn.DataParallel(self.resnet)
            self.densenet = nn.DataParallel(self.densenet)
            self.model = nn.DataParallel(self.model)
            
        self.roberta.to(device)
        self.resnet.to(device)
        self.densenet.to(device)
        self.model.to(device)
        
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
        criterion = nn.CrossEntropyLoss()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(params, lr=self.opt.learning_rate)

        global_step = 0

        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print(f'epoch: {epoch}')
            epoch_start_time = time.time()

            self.model.train()
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                batch_start_time = time.time()

                optimizer.zero_grad()
                input_ids_text = sample_batched['input_ids_text'].to(device)
                attention_mask_text = sample_batched['attention_mask_text'].to(device)
                input_ids_topic = sample_batched['input_ids_topic'].to(device)
                attention_mask_topic = sample_batched['attention_mask_topic'].to(device)
                images = sample_batched['image'].to(device)
                targets = sample_batched['polarity'].to(device)

                resnet_features = self.resnet(images)
                densenet_features = self.densenet(images)

                roberta_inputs_text = {
                    'input_ids': input_ids_text,
                    'attention_mask': attention_mask_text
                }
                roberta_text_features = self.roberta(**roberta_inputs_text).last_hidden_state[:, 0, :]

                roberta_inputs_topic = {
                    'input_ids': input_ids_topic,
                    'attention_mask': attention_mask_topic
                }
                roberta_topic_features = self.roberta(**roberta_inputs_topic).last_hidden_state[:, 0, :]

                outputs = self.model(roberta_text_features, roberta_topic_features, 
                                     resnet_features, densenet_features, images)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.clip_grad)
                optimizer.step()

                global_step += 1
                self.writer.add_scalar('Loss/train', loss.item(), global_step)

                batch_end_time = time.time()

                if i_batch % self.opt.log_step == 0:
                    dev_acc, dev_f1, dev_loss = self.evaluate(self.dev_data_loader)
                    test_acc, test_f1, test_loss = self.evaluate(self.test_data_loader)

                    self.writer.add_scalar('Loss/val', dev_loss, global_step)

                    print(f'Batch {i_batch} completed in {batch_end_time - batch_start_time:.2f} seconds '
                          f'({(batch_end_time - batch_start_time) / 60:.2f} minutes)')

                    if dev_f1 > self.max_dev_f1:
                        print(f"New best dev_f1: {dev_f1:.6f} (previous best: {self.max_dev_f1:.6f})")
                        self.max_dev_f1 = dev_f1
                        self.max_test_f1 = test_f1

                    print(f'loss: {loss.item():.6f}, dev_acc: {dev_acc * 100:.2f}% ({dev_acc:.6f}), '
                          f'dev_f1: {dev_f1 * 100:.2f}% ({dev_f1:.6f}), test_acc: {test_acc * 100:.2f}% '
                          f'({test_acc:.6f}), test_f1: {test_f1 * 100:.2f}% ({test_f1:.6f})')

            epoch_end_time = time.time()
            print(f'Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds '
                  f'({(epoch_end_time - epoch_start_time) / 60:.2f} minutes)')

        # Flush the SummaryWriter to ensure all logs are saved to disk
        self.writer.flush()

        print(f"RESULT: Max Dev F1: {self.max_dev_f1:.6f}, Max Test F1: {self.max_test_f1:.6f}")

    def evaluate(self, data_loader):
        self.model.eval()
        true_labels, pred_probs = [], []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_loader):
                input_ids_text = sample_batched['input_ids_text'].to(device)
                attention_mask_text = sample_batched['attention_mask_text'].to(device)
                input_ids_topic = sample_batched['input_ids_topic'].to(device)
                attention_mask_topic = sample_batched['attention_mask_topic'].to(device)
                images = sample_batched['image'].to(device)
                targets = sample_batched['polarity'].to(device)

                resnet_features = self.resnet(images)
                densenet_features = self.densenet(images)

                roberta_inputs_text = {
                    'input_ids': input_ids_text,
                    'attention_mask': attention_mask_text
                }
                roberta_text_features = self.roberta(**roberta_inputs_text).last_hidden_state[:, 0, :]

                roberta_inputs_topic = {
                    'input_ids': input_ids_topic,
                    'attention_mask': attention_mask_topic
                }
                roberta_topic_features = self.roberta(**roberta_inputs_topic).last_hidden_state[:, 0, :]

                outputs = self.model(roberta_text_features, roberta_topic_features, 
                                     resnet_features, densenet_features, images)

                loss = criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)

                true_labels.append(targets.cpu().numpy())
                pred_probs.append(outputs.cpu().numpy())

        true_labels = np.concatenate(true_labels)
        pred_probs = np.concatenate(pred_probs)

        f1_score = macro_f1(true_labels, pred_probs)
        accuracy = (np.argmax(pred_probs, axis=-1) == true_labels).mean()
        avg_loss = total_loss / len(data_loader.dataset)
        return accuracy, f1_score, avg_loss
      
    def read_tensorboard_loss(self, log_dir=None):
        print('Reading TensorBoard loss at each epoch:')
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        print('Available tags:', ea.Tags())
        
        loss_events = ea.Scalars('Loss/train')
        for event in loss_events:
            print(f"Step: {event.step}, Loss: {event.value}")

    def plot_tensorboard_loss(self, log_dir=None):
        output_file = os.path.join(log_dir, 'trainval_loss_curves.png')
        print(f"Output File: {output_file}")

        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        train_loss_events = ea.Scalars('Loss/train')
        val_loss_events = ea.Scalars('Loss/val')

        train_steps = [event.step for event in train_loss_events]
        train_losses = [event.value for event in train_loss_events]
        val_steps = [event.step for event in val_loss_events]
        val_losses = [event.value for event in val_loss_events]

        plt.figure(figsize=(12, 6))
        plt.plot(train_steps, train_losses, label='Training Loss', alpha=0.7)
        plt.plot(val_steps, val_losses, label='Validation Loss', color='orange', alpha=0.7)

        # Calculate steps where epochs end
        num_batches_per_epoch = len(self.train_data_loader)
        epoch_end_steps = [num_batches_per_epoch * (epoch + 1) for epoch in range(self.opt.num_epoch)]

        # Add vertical lines at epoch boundaries
        for idx, epoch_step in enumerate(epoch_end_steps):
            plt.axvline(x=epoch_step, color='gray', linestyle='--', linewidth=0.8)
            plt.text(epoch_step, max(train_losses)*0.95, f'Epoch {idx+1}', rotation=90,
                    verticalalignment='top', fontsize=8, color='gray')

        plt.xlabel('Training Steps (Batches)')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Steps')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file)
        print(f'Training and validation loss curves saved to {output_file}')