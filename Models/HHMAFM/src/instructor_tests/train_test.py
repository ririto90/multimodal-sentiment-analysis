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

# This function calculates macro-averaged precision, recall, and F1-score 
# for true labels and predicted probabilities.
def macro_f1(y_true, y_pred):
    # Get predicted class labels by finding the index of the max probability
    preds = np.argmax(y_pred, axis=-1)
    
    # Store true labels
    true = y_true
    
    # Calculate macro-averaged precision, recall, F1-score, and support
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(true, preds, average='macro')
    
    # Return the calculated metrics
    return p_macro, r_macro, f_macro
  
class Instructor:
    def __init__(self, opt):
        # Store provided options
        self.opt = opt
        
        # Iterate over all arguments in opt and print
        print('> training arguments:')
        for arg in vars(opt):
            print('>>> {0}: {1}'.format(arg, getattr(opt, arg)))
            
        # Define a sequence of transformations to be applied to the input images
        transform = transforms.Compose([
            
            # Randomly crops the image to a specified size
            transforms.RandomCrop(opt.crop_size),  # (crop_size default = 224x224)
            
            # Randomly flip the image horizontally
            transforms.RandomHorizontalFlip(),
            
            # Convert the image to a tensor (required for PyTorch models)
            transforms.ToTensor(),
            
            # Normalize the image tensor
            transforms.Normalize((0.485, 0.456, 0.406),  # Mean for each channel (R, G, B)
                                (0.229, 0.224, 0.225))  # Standard deviation for each channel (R, G, B)
        ])
            
        # Initialize the ABSADatesetReader
        absa_dataset = MVSADatasetReader(transform, dataset=opt.dataset, max_seq_len=opt.max_seq_len, path_image=opt.path_image)
        
        # Create DataLoaders for the train, test, and dev dataset
        self.train_data_loader = DataLoader(dataset=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.dev_data_loader = DataLoader(dataset=absa_dataset.dev_data, batch_size=opt.batch_size, shuffle=False)
        self.test_data_loader = DataLoader(dataset=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False)
    
        print('building model')
        
        # Instantiate and load pre-trained weights for ResNet-152
        # net = getattr(resnet, 'resnet152')()  
        # net.load_state_dict(torch.load(os.path.join(opt.resnet_root, 'resnet152.pth')))
        
        # Initialize a custom ResNet encoder & move to (GPU/CPU)
        # self.encoder = myResnet(net, opt.fine_tune_cnn, self.opt.device).to(device)
        
        # RoBERTa - textual and topic feature extraction
        self.roberta = RobertaModel.from_pretrained('roberta-base').to(device)
        
        # Resnet-152 - low-level features
        self.resnet = models.resnet152(pretrained=True).to(device)
        
        # DenseNet high-level features
        self.densenet = models.densenet121(pretrained=True).to(device)
        
        # Initialize main model with embedding matrix from the dataset and other options & move to (GPU/CPU)
        self.model = opt.model_class(opt, self.roberta_model, self.resnet, self.densenet).to(device)
        
        # re-initializing weights
        self.reset_parameters()
        # This method resets the parameters of the model
        # Counts the number of trainable and non-trainable parameters.
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
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        
    # def extract_text_features(self, text_indices):
    #     inputs = torch.tensor(text_indices).unsqueeze(0).to(device)  # Add batch dimension and move to device
    #     outputs = self.roberta_model(inputs)
    #     return outputs.last_hidden_state[:, 0, :]
          
    def run(self):
        # Define the loss function as CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
        
        # Filter the model parameters to include only those that require gradients (i.e., trainable parameters)
        text_params = filter(lambda p: p.requires_grad, self.model.parameters())

        # If the fine-tuning option for the CNN is enabled
        if self.opt.fine_tune_cnn:
            # Combine the text model parameters and the encoder (CNN) parameters into a single list
            params = list(text_params) + list(self.encoder.parameters())
        else:
            # If fine-tuning is not enabled, print a message indicating that only text parameters are included
            print('parameters only include text parts without word embeddings')
            # Use only the text model parameters, excluding the CNN parameters
            params = list(text_params)
            
        # Initialize the optimizer with the selected parameters & learning rate    
        optimizer = self.opt.optimizer(params, lr=self.opt.learning_rate)
        
        max_dev_acc = 0
        max_dev_p = 0
        max_dev_r = 0
        max_dev_f1 = 0
        max_test_p = 0
        max_test_r = 0
        max_test_f1 = 0
        max_test_acc = 0
        global_step = 0
        track_list = list()
        
        for epoch in range(self.opt.num_epoch):
        
            # Print Epoch
            print('>' * 100)
            print('epoch: ', epoch)
            
            # Initialize counters
            n_correct, n_total = 0, 0
            
            # Loop over the batches of data from the training dataset
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                # print("Sample keys:", sample_batched.keys())
            
                # Increment the global step
                global_step += 1

                # Switch the encoder and model to training mode
                # This is crucial because some layers, like dropout and batch normalization, behave differently during training versus evaluation
                self.roberta.train()
                self.resnet.train()
                self.densenet.train()
                self.model.train()
                
                # Clear the gradient accumulators in the optimizer and encoder
                # Gradients accumulate by default in PyTorch. Before processing each new batch
                # we clear the gradients to avoid incorrent updates during backpropagation
                optimizer.zero_grad()
                # self.encoder.zero_grad()
                
                # Move features, images, and targets to (GPU/CPU)
                input_ids_text = sample_batched['input_ids_text'].to(device)
                attention_mask_text = sample_batched['attention_mask_text'].to(device)
                input_ids_topic = sample_batched['input_ids_topic'].to(device)
                attention_mask_topic = sample_batched['attention_mask_topic'].to(device)
                targets = sample_batched['polarity'].to(device)
                images = sample_batched['image'].to(device)
                
                # Extract features from images with the pretrained CNN resnet
                # imgs_f: Feature representations of the images.
                # img_mean: The mean feature vector of the images.
                # img_att: Attention weights or similar features from the images.
                with torch.no_grad():
                    # imgs_f, img_mean, img_att = self.encoder(images)
                    resnet_features = self.resnet(images)
                    densenet_features = self.densenet(images)
                    
                    # Extract features from RoBERTa for text and topic
                    roberta_inputs_text = {
                        'input_ids': input_ids_text,
                        'attention_mask': attention_mask_text
                    }
                    roberta_text_features = self.roberta_model(**roberta_inputs_text).last_hidden_state[:, 0, :]
                    
                    roberta_inputs_topic = {
                        'input_ids': input_ids_topic,
                        'attention_mask': attention_mask_topic
                    }
                    roberta_topic_features = self.roberta_model(**roberta_inputs_topic).last_hidden_state[:, 0, :]
                    
                
                # Pass input and image data through the Main Model
                outputs = self.model(roberta_text_features, roberta_topic_features, resnet_features, densenet_features)

                # Calculates how far the model's predictions (outputs) are from the actual labels (targets)
                loss = criterion(outputs, targets)
                
                # Computes the gradients of the loss with respect to each parameter of the model.
                # These gradients are then used to update the model’s parameters during optimization
                loss.backward()
                
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(params, self.opt.clip_grad)
                    
                # Updates the model's parameters using the gradients computed during backpropagation.
                optimizer.step()
                
                if global_step % self.opt.log_step == 0:
                
                    # switches the model and encoder to evaluation mode.
                    # In evaluation mode, layers like dropout and batch normalization behave differently
                    self.encoder.eval()
                    self.model.eval()
                    
                    # Track the number of correct predictions and the total number of samples for the train, dev, and test datasets.
                    n_train_correct, n_train_total = 0, 0
                    n_dev_correct, n_dev_total = 0, 0
                    n_test_correct, n_test_total = 0, 0
                    
                    with torch.no_grad():
                        # Evaluate on the training data
                        for t_batch, t_sample_batched in enumerate(self.train_data_loader):
                            t_inputs = t_sample_batched['text_indices'].to(device)
                            t_targets = t_sample_batched['polarity'].to(device)
                            t_images = t_sample_batched['image'].to(device)
                        
                            # Extract features from images with the pretrained CNN resnet
                            t_imgs_f, t_img_mean, t_img_att = self.encoder(t_images)
                        
                            # Pass input and image data through the Main Model
                            t_outputs = self.model([t_inputs], t_imgs_f)
                            n_train_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                            n_train_total += len(t_outputs)
                        train_acc = n_train_correct / n_train_total

                        # Evaluate on the validation (dev) data
                        true_label_list = []
                        pred_label_list = []
                        for t_batch, t_sample_batched in enumerate(self.dev_data_loader):
                            t_inputs = t_sample_batched['text_indices'].to(device)
                            t_targets = t_sample_batched['polarity'].to(device)
                            t_images = t_sample_batched['image'].to(device)
                        
                            # Extract features from images with the pretrained CNN resnet
                            t_imgs_f, t_img_mean, t_img_att = self.encoder(t_images)
                        
                            # Pass input and image data through the Main Model
                            t_outputs = self.model([t_inputs], t_imgs_f)
                            n_dev_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                            n_dev_total += len(t_outputs)

                            true_label_list.append(t_targets.detach().cpu())
                            pred_label_list.append(t_outputs.detach().cpu())

                        true_label = np.concatenate(true_label_list)
                        pred_outputs = np.concatenate(pred_label_list)
                            
                        dev_p, dev_r, dev_f1 = macro_f1(true_label, pred_outputs)
                        dev_acc = n_dev_correct / n_dev_total

                        # Evaluate on the test data
                        true_label_list = []
                        pred_label_list = []
                        for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                            t_inputs = t_sample_batched['text_indices'].to(device)
                            t_targets = t_sample_batched['polarity'].to(device)
                            t_images = t_sample_batched['image'].to(device)
                        
                            # Extract features from images with the pretrained CNN resnet
                            t_imgs_f, t_img_mean, t_img_att = self.encoder(t_images)
                        
                            # Pass input and image data through the Main Model
                            t_outputs = self.model([t_inputs], t_imgs_f)
                            n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                            n_test_total += len(t_outputs)

                            true_label_list.append(t_targets.detach().cpu())
                            pred_label_list.append(t_outputs.detach().cpu())

                        true_label = np.concatenate(true_label_list)
                        pred_outputs = np.concatenate(pred_label_list)
                        
                        test_p, test_r, test_f1  = macro_f1(true_label, pred_outputs)
                        test_acc = n_test_correct / n_test_total
                        # if test_acc > max_test_acc:
                        # if dev_acc > max_dev_acc:
                        # if dev_acc+dev_f1 > max_dev_acc+max_dev_f1:
                        if dev_acc > max_dev_acc:
                            max_dev_acc = dev_acc
                            max_dev_p = dev_p
                            max_dev_r = dev_r
                            max_dev_f1 = dev_f1
                            max_test_acc = test_acc
                            max_test_p = test_p
                            max_test_r = test_r
                            max_test_f1 = test_f1
                            
                            track_list.append(
                                {'loss': loss.item(), 'dev_acc': dev_acc, 'dev_p': dev_p,'dev_r': dev_r,'dev_f1': dev_f1, \
                                    'test_acc': test_acc, 'test_p': test_p,'test_r': test_r,'test_f1': test_f1})

                        print('loss: {:.4f}, acc: {:.4f}, dev_acc: {:.4f}, test_acc: {:.4f}'.format(loss.item(),\
                                train_acc, dev_acc, test_acc))
                            
                        print('dev_f1: {:.4f}, test_f1: {:.4f}'.format(dev_f1, test_f1))
                            
        print('max_dev_acc: {0}, test_acc: {1}'.format(max_dev_acc, max_test_acc))
        print('dev_p: {0}, dev_r: {1}, dev_f1: {2}, test_p: {3}, test_r: {4}, test_f1: {5}'.format(max_dev_p,\
                        max_dev_r, max_dev_f1, max_test_p, max_test_r, max_test_f1))

if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rand_seed', default=8, type=int)
    parser.add_argument('--model_name', default='mmfusion', type=str)
    parser.add_argument('--dataset', default='mvsa-mts-100', type=str, help='twitter, snap')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--num_epoch', default=8, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--log_step', default=1, type=int) # 50
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--max_seq_len', default=64, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--att_file', default='./att_file/', help='path of attention file')
    parser.add_argument('--pred_file', default='./pred_file/', help='path of prediction file')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='grad clip at')
    #parser.add_argument('--path_image', default='../../../visual_attention_ner/twitter_subimages', help='path to images')
    parser.add_argument('--path_image', default='./twitter_subimages', help='path to images')
    #parser.add_argument('--path_image', default='./twitter15_images', help='path to images')
    #parser.add_argument('--path_image', default='./snap_subimages', help='path to images')
    parser.add_argument('--crop_size', type=int, default = 224, help='crop size of image')
    parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained CNN if True')
    parser.add_argument('--att_mode', choices=['text', 'vis_only', 'vis_concat',  'vis_att', 'vis_concat_attimg', \
                                               'text_vis_att_img_gate', 'vis_att_concat', 'vis_att_attimg', \
    'vis_att_img_gate', 'vis_concat_attimg_gate'], default ='vis_concat_attimg_gate', \
    help='different attention mechanism')
    parser.add_argument('--resnet_root', default='../util_models/resnet', help='path the pre-trained cnn models')
    parser.add_argument('--checkpoint', default='./checkpoint/', help='path to checkpoint prefix')
    parser.add_argument('--load_check_point', action='store_true', help='path of checkpoint')
    parser.add_argument('--load_opt', action='store_true', help='load optimizer from ')
    parser.add_argument('--tfn', action='store_true', help='whether to use TFN')
    
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    if opt.dataset == "mvsa-mts":
        opt.path_image = "../../Datasets/MVSA-Modified/images-indexed"
        opt.max_seq_len = 25
        opt.rand_seed = 28
    elif opt.dataset == "mvsa-mts-100":
        opt.path_image = "../../Datasets/MVSA-Modified/images-indexed"
        opt.max_seq_len = 40
        opt.rand_seed = 25
    else:
        print("The dataset name is not right!")

    if opt.tfn:
        print("************add another tfn layer*************")
    else:
        print("************no tfn layer************")

    random.seed(opt.rand_seed)
    np.random.seed(opt.rand_seed)
    torch.manual_seed(opt.rand_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(opt.rand_seed)

    model_classes = {
        'mmfusion': MMFUSION
    }
    fusion_inputs = {
        'mmfusion': ['input_ids_text','attention_mask_text', 'input_ids_topic', 'attention_mask_topic'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.feature_columns = fusion_inputs[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = device

    ins = Instructor(opt)
    ins.run()
