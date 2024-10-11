from util_tests.data_utils_test import MVSADatasetReader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

import argparse
from transformers import RobertaModel
import os
import random
import matplotlib.pyplot as plt

from torchvision import transforms
from models.mmfusion import MMFUSION

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Macro F1 Score Calculation
def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, preds, average='macro')
    return p_macro, r_macro, f_macro

class Instructor:
    def __init__(self, opt):
        self.opt = opt
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
        
        self.train_data_loader = DataLoader(dataset=mvsa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.dev_data_loader = DataLoader(dataset=mvsa_dataset.dev_data, batch_size=opt.batch_size, shuffle=False)
        self.test_data_loader = DataLoader(dataset=mvsa_dataset.test_data, batch_size=opt.batch_size, shuffle=False)
    
        print('building model')

        self.roberta = RobertaModel.from_pretrained('roberta-base').to(device)
        self.resnet = models.resnet152(pretrained=True).to(device)
        self.densenet = models.densenet121(pretrained=True).to(device)
        
        self.model = opt.model_class(opt).to(device)
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
            
            self.model.train()
            for i_batch, sample_batched in enumerate(self.train_data_loader):
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

                outputs = self.model(roberta_text_features, roberta_topic_features, resnet_features, densenet_features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                if i_batch % self.opt.log_step == 0:
                    dev_acc, dev_f1 = self.evaluate(self.dev_data_loader)
                    test_acc, test_f1 = self.evaluate(self.test_data_loader)
                    
                    if dev_f1 > max_dev_f1:
                        max_dev_f1 = dev_f1
                        max_test_f1 = test_f1

                    print(f'loss: {loss.item()}, dev_acc: {dev_acc}, dev_f1: {dev_f1}, test_acc: {test_acc}, test_f1: {test_f1}')

        print(f'Max dev F1: {max_dev_f1}, Max test F1: {max_test_f1}')
        
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rand_seed', default=8, type=int)
    parser.add_argument('--model_name', default='mmfusion', type=str)
    parser.add_argument('--dataset', default='mvsa-mts-100', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--num_epoch', default=8, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--max_seq_len', default=64, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    parser.add_argument('--path_image', default='./twitter_subimages', help='path to images')
    parser.add_argument('--crop_size', type=int, default=224)

    opt = parser.parse_args()

    random.seed(opt.rand_seed)
    np.random.seed(opt.rand_seed)
    torch.manual_seed(opt.rand_seed)

    model_classes = {
        'mmfusion': MMFUSION
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
