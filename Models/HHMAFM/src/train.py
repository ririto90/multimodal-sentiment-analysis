# Instructor Class

import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from transformers import RobertaModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

from data_utils import MVSADatasetReader
from models.mmfusion import MMFUSION
from models.cmhafusion import CMHAFUSION
from models.mfcchfusion import MFCCHFUSION
from models.mfcchfusion2 import MFCCHFUSION2

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Number of GPUs available: {torch.cuda.device_count()}')

log_dir = os.getenv('NEW_LOGS_DIR')
if log_dir is None:
    raise ValueError("NEW_LOGS_DIR environment variable is not set")
print(f"Logs directory: {log_dir}")

def print_features(input):
    print(input)

def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, preds, average='macro')
    return p_macro, r_macro, f_macro

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        print(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        print('> training arguments:')
        for arg in vars(opt):
            print(f'>>> {arg}: {getattr(opt, arg)}')
            
        transform = transforms.Compose([
            transforms.RandomCrop(opt.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
            
        mvsa_dataset = MVSADatasetReader(transform, dataset=opt.dataset, max_seq_len=opt.max_seq_len, path_image=opt.path_image)
        opt.num_classes = mvsa_dataset.num_classes
        
        self.train_data_loader = DataLoader(dataset=mvsa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.dev_data_loader = DataLoader(dataset=mvsa_dataset.dev_data, batch_size=opt.batch_size, shuffle=False)
        self.test_data_loader = DataLoader(dataset=mvsa_dataset.test_data, batch_size=opt.batch_size, shuffle=False)
    
        print('building model')

        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.resnet = models.resnet152(pretrained=True)
        self.densenet = models.densenet121(pretrained=True)
        self.model = opt.model_class(opt)
        
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

        max_dev_f1, max_test_f1 = 0, 0
        
        for epoch in range(self.opt.num_epoch):
            # total_loss = 0.0
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
                
                # if i_batch == 1:
                #         print_features(input_ids_text)
                #         print_features(attention_mask_text)
                #         print_features(input_ids_topic)
                #         print_features(attention_mask_topic)
                
                
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

                outputs = self.model(roberta_text_features, roberta_topic_features, resnet_features, densenet_features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # epoch_loss += loss.item()
                
                fractional_epoch = epoch + i_batch / len(self.train_data_loader)
                self.writer.add_scalar('Loss/train', loss.item(), fractional_epoch)
                batch_end_time = time.time()
                
                if i_batch % self.opt.log_step == 0:
                    dev_acc, dev_f1, dev_loss = self.evaluate(self.dev_data_loader)
                    test_acc, test_f1, test_loss = self.evaluate(self.test_data_loader)
                    
                    self.writer.add_scalar('Loss/val', dev_loss, fractional_epoch)
                    print(f'Batch {i_batch} completed in {batch_end_time - batch_start_time:.2f} seconds ({(batch_end_time - batch_start_time) / 60:.2f} minutes)')
                    
                    if dev_f1 > max_dev_f1:
                        print(f"max_dev_f1: {max_dev_f1}, max_test_f1: {max_test_f1}")
                        max_dev_f1 = dev_f1
                        max_test_f1 = test_f1

                    print(f'loss: {loss.item():.6f}, dev_acc: {dev_acc * 100:.2f}% ({dev_acc:.6f}), dev_f1: {dev_f1 * 100:.2f}% ({dev_f1:.6f}), test_acc: {test_acc * 100:.2f}% ({test_acc:.6f}), test_f1: {test_f1 * 100:.2f}% ({test_f1:.6f})')

            epoch_end_time = time.time()
            # avg_epoch_loss = epoch_loss / len(self.train_data_loader)
            # self.writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)
            print(f'Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds ({(epoch_end_time - epoch_start_time) / 60:.2f} minutes)')
            
        # Flush the SummaryWriter to ensure all logs are saved to disk
        self.writer.flush()
            
        print(f'Max dev F1: {max_dev_f1 * 100:.2f}% ({max_dev_f1:.6f}), Max test F1: {max_test_f1 * 100:.2f}% ({max_test_f1:.6f})')
        
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

                outputs = self.model(roberta_text_features, roberta_topic_features, resnet_features, densenet_features)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item() * targets.size(0)
                
                true_labels.append(targets.cpu().numpy())
                pred_probs.append(outputs.cpu().numpy())
        
        true_labels = np.concatenate(true_labels)
        pred_probs = np.concatenate(pred_probs)

        precision, recall, f1_score = macro_f1(true_labels, pred_probs)
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
        
        plt.figure()
        plt.plot(train_steps, train_losses, label='Training Loss')
        plt.plot(val_steps, val_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.savefig(output_file)
        print(f'Training and validation loss curves saved to {output_file}')

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rand_seed', default=8, type=int)
    parser.add_argument('--model_name', default='mmfusion', type=str)
    parser.add_argument('--dataset', default='mvsa-mts-100', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--num_epoch', default=8, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--max_seq_len', default=64, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    parser.add_argument('--path_image', default='./Datasets/MVSA-MTS/images-indexed', help='path to images')
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--n_heads', type=int, default=8)
    
    parser.add_argument('--roberta_text_feature_dim', type=int, default=768)
    parser.add_argument('--roberta_topic_feature_dim', type=int, default=50)
    parser.add_argument('--resnet_feature_dim', type=int, default=2048)
    parser.add_argument('--densenet_feature_dim', type=int, default=1024)
    
    parser.add_argument('--common_dim', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=3)

    opt = parser.parse_args()

    random.seed(opt.rand_seed)
    np.random.seed(opt.rand_seed)
    torch.manual_seed(opt.rand_seed)

    model_classes = {
        'mmfusion': MMFUSION,
        'cmhafusion': CMHAFUSION,
        'mfcchfusion': MFCCHFUSION,
        'mfcchfusion2': MFCCHFUSION2,
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    optimizers = {
        'adam': torch.optim.Adam,
        'adadelta': torch.optim.Adadelta,
        'sgd': torch.optim.SGD,
    }

    opt.model_class = model_classes[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    ins = Instructor(opt)
    ins.run()
    time.sleep(10)
    
    ins.read_tensorboard_loss(log_dir)
    ins.plot_tensorboard_loss(log_dir=log_dir)
    
    end_time = time.time()
    print(f"Total Completion Time: {(end_time - start_time) / 60:.2f} minutes. ({(end_time - start_time) / 3600:.2f} hours) ")
