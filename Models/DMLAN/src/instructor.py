# instructor.py

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, Inception_V3_Weights
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

from data_utils import MVSADatasetReader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

        # Data transformations (adjusted for Inception V3)
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])

        # Load dataset
        mvsa_dataset = MVSADatasetReader(transform, dataset=opt.dataset, 
                                         max_seq_len=opt.max_seq_len)
        opt.num_classes = mvsa_dataset.num_classes

        self.train_data_loader = DataLoader(dataset=mvsa_dataset.train_data, 
                                            batch_size=opt.batch_size, shuffle=True)
        self.val_data_loader = DataLoader(dataset=mvsa_dataset.val_data, 
                                          batch_size=opt.batch_size, shuffle=False)
        self.test_data_loader = DataLoader(dataset=mvsa_dataset.test_data, 
                                           batch_size=opt.batch_size, shuffle=False)

        print('Building model')

        # Add embedding layer initialized with GloVe embeddings
        embedding_matrix = torch.tensor(mvsa_dataset.embedding_matrix, dtype=torch.float)
        num_embeddings, embedding_dim = embedding_matrix.size()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=1)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False 

        # Add LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=768, 
                            num_layers=opt.num_layers, batch_first=True, bidirectional=True)

        # Initialize Inception V3 for images
        self.inception = inception_v3(
            weights=Inception_V3_Weights.DEFAULT,
            aux_logits=True
        )
        self.inception.Mixed_7c.register_forward_hook(self.save_feature_maps)
        self.inception.fc = nn.Identity()  # Remove final classification layer
        self.feature_maps = None 

        # Freeze Inception V3 parameters
        for param in self.inception.parameters():
            param.requires_grad = False

        self.model = opt.model_class(opt)
        self.opt.counter += 1
        if self.opt.counter < 3:
            print(self.opt.counter)


        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            self.embedding = nn.DataParallel(self.embedding)
            self.lstm = nn.DataParallel(self.lstm)
            self.inception = nn.DataParallel(self.inception)
            self.model = nn.DataParallel(self.model)

        self.embedding.to(device)
        self.lstm.to(device)
        self.inception.to(device)
        self.model.to(device)

        self.reset_parameters()

    def save_feature_maps(self, module, input, output):
        # Hook to save the feature maps from a specific layer
        self.feature_maps = output  # Save the output of Mixed_7c layer

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
        if self.opt.weight_decay is not None and self.opt.weight_decay > 0:
            optimizer = self.opt.optimizer(params, lr=self.opt.learning_rate, weight_decay=self.opt.weight_decay)
            print("Using weight decay")
        else:
            optimizer = self.opt.optimizer(params, lr=self.opt.learning_rate)
            print("No weight decay")

        global_step = 0

        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print(f'epoch: {epoch}')
            epoch_start_time = time.time()

            self.model.train()
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                batch_start_time = time.time()

                optimizer.zero_grad()
                text_indices = sample_batched['text_indices'].to(device)
                images = sample_batched['image'].to(device)
                targets = sample_batched['polarity'].to(device).long()
                
                if self.opt.counter == 0:
                    print("targets.shape:", targets.shape, "targets.dtype:", targets.dtype)

                # Text processing through embedding and LSTM
                embedded_text = self.embedding(text_indices)
                lstm_output, (h_n, c_n) = self.lstm(embedded_text)
                print(f"LSTM output shape: {lstm_output.shape}")
                text_features = lstm_output[:, -1, :]
                print(f"text_features: {text_features.shape}")

                # Image processing through Inception V3
                self.feature_maps = None  # Reset feature maps
                _ = self.inception(images)  # Forward pass to trigger hook
                image_features = self.feature_maps  # Feature maps from Mixed_7c layer

                outputs = self.model(text_features, image_features)
                
                if self.opt.counter == 1:
                    print("outputs.shape:", outputs.shape)
                    print("outputs.dtype:", outputs.dtype)
                self.opt.counter += 1
                if self.opt.counter < 3:
                    print(self.opt.counter)
                
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opt.clip_grad)
                optimizer.step()

                global_step += 1
                self.writer.add_scalar('Loss/train', loss.item(), global_step)

                batch_end_time = time.time()

                if i_batch % self.opt.log_step == 0:
                    val_acc, val_f1, val_loss = self.evaluate(self.val_data_loader)
                    test_acc, test_f1, test_loss = self.evaluate(self.test_data_loader)

                    self.writer.add_scalar('Loss/val', val_loss, global_step)

                    print(f'Batch {i_batch} completed in {batch_end_time - batch_start_time:.2f} seconds '
                          f'({(batch_end_time - batch_start_time) / 60:.2f} minutes)')

                    if val_f1 > self.max_val_f1:
                        print(f"New best val_f1: {val_f1:.6f} (previous best: {self.max_val_f1:.6f})")
                        self.max_val_f1 = val_f1
                        self.max_test_f1 = test_f1

                    print(f'loss: {loss.item():.6f}, val_acc: {val_acc * 100:.2f}% ({val_acc:.6f}), '
                          f'val_f1: {val_f1 * 100:.2f}% ({val_f1:.6f}), test_acc: {test_acc * 100:.2f}% '
                          f'({test_acc:.6f}), test_f1: {test_f1 * 100:.2f}% ({test_f1:.6f})')

            epoch_end_time = time.time()
            print(f'Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds '
                  f'({(epoch_end_time - epoch_start_time) / 60:.2f} minutes)')

        # Flush the SummaryWriter to ensure all logs are saved to disk
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
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_loader):
                text_indices = sample_batched['text_indices'].to(device)
                images = sample_batched['image'].to(device)
                targets = sample_batched['polarity'].to(device).long()

                # Text processing
                embedded_text = self.embedding(text_indices)
                lstm_output, (h_n, c_n) = self.lstm(embedded_text)
                text_features = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)

                # Image processing
                self.feature_maps = None  # Reset feature maps
                _ = self.inception(images)
                image_features = self.feature_maps

                outputs = self.model(text_features, image_features)

                loss = criterion(outputs, targets)
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

        loss_events = ea.Scalars('Loss/train')

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

    def plot_confusion_matrix(self, true_labels, pred_labels, classes, log_dir):
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot using seaborn heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Normalized Confusion Matrix')

        # Save the figure
        output_file = os.path.join(log_dir, 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f'Confusion matrix saved to {output_file}')
