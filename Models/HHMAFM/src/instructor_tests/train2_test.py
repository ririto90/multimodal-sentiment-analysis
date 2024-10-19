from util_tests.data_utils_test import MVSADatasetReader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import time

import argparse
from transformers import RobertaModel
import os
import random
import matplotlib.pyplot as plt

from torchvision import transforms
from models.mmfusion import MMFUSION
from models.cmhafusion import CMHAFUSION
from models.mfcchfusion import MFCCHFUSION

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Number of GPUs available: {torch.cuda.device_count()}')

def print_features(input):
    print(input)

def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, preds, average='macro')
    return p_macro, r_macro, f_macro

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        self.train_losses = []
        
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
                
                if i_batch == 1:
                        print_features(input_ids_text)
                        print_features(attention_mask_text)
                        print_features(input_ids_topic)
                        print_features(attention_mask_topic)
                
                
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
                
                self.train_losses.append(loss.item())
                
                batch_end_time = time.time()
                
                
                if i_batch % self.opt.log_step == 0:
                    print(f'Batch {i_batch} completed in {batch_end_time - batch_start_time:.2f} seconds ({(batch_end_time - batch_start_time) / 60:.2f} minutes)')
                    dev_acc, dev_f1 = self.evaluate(self.dev_data_loader)
                    test_acc, test_f1 = self.evaluate(self.test_data_loader)
                    
                    if dev_f1 > max_dev_f1:
                        print(f"max_dev_f1: {max_dev_f1}, max_test_f1: {max_test_f1}")
                        max_dev_f1 = dev_f1
                        max_test_f1 = test_f1

                    print(f'loss: {loss.item():.6f}, dev_acc: {dev_acc * 100:.2f}% ({dev_acc:.6f}), dev_f1: {dev_f1 * 100:.2f}% ({dev_f1:.6f}), test_acc: {test_acc * 100:.2f}% ({test_acc:.6f}), test_f1: {test_f1 * 100:.2f}% ({test_f1:.6f})')

            epoch_end_time = time.time()
            print(f'Epoch {epoch} completed in {epoch_end_time - epoch_start_time:.2f} seconds ({(epoch_end_time - epoch_start_time) / 60:.2f} minutes)')
            
        print(f'Max dev F1: {max_dev_f1 * 100:.2f}% ({max_dev_f1:.6f}), Max test F1: {max_test_f1 * 100:.2f}% ({max_test_f1:.6f})')
        self.save_training_loss_plot()
        
    def evaluate(self, data_loader):
        self.model.eval()
        true_labels, pred_probs = [], []
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
                
                true_labels.append(targets.cpu().numpy())
                pred_probs.append(outputs.cpu().numpy())
        
        true_labels = np.concatenate(true_labels)
        pred_probs = np.concatenate(pred_probs)

        precision, recall, f1_score = macro_f1(true_labels, pred_probs)
        accuracy = (np.argmax(pred_probs, axis=-1) == true_labels).mean()

        return accuracy, f1_score
    def save_training_loss_plot(self):
        # Get the log directory from the environment variable
        log_dir = os.getenv('NEW_LOG_DIR')
        if log_dir:
            log_dir = os.path.dirname(log_dir)
            plot_path = os.path.join(log_dir, 'training_loss_curve.png')
            
            # Plot the training loss curve
            plt.figure()
            plt.plot(self.train_losses, label='Training Loss')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.legend()
            plt.savefig(plot_path)
            print(f'Training loss curve saved to {plot_path}')
        else:
            print('Environment variable SLURM_JOB_OUTPUT not set. Cannot save training loss plot.')

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
    
    end_time = time.time()
    print(f"Total Completion Time: {(end_time - start_time) / 60:.2f}")
