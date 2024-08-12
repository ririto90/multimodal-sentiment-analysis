import os
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms

from utils.data import ABSADatesetReader

import resnet.resnet as resnet
from resnet.resnet_utils import myResnet

from models.mmfusion import MMFUSION

from sklearn.metrics import precision_recall_fscore_support

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='mmtdtan', type=str)
    parser.add_argument('--dataset', default='twitter', type=str, help='twitter, snap')
    parser.add_argument('--rand_seed', default=8, type=int)
    parser.add_argument('--num_epoch', default=8, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--log_step', default=50, type=int)

    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--dropout_rate', default=0.5, type=float)

    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--max_seq_len', default=64, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--att_file', default='./att_file/', help='path of attention file')
    parser.add_argument('--pred_file', default='./pred_file/', help='path of prediction file')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='grad clip at')
    parser.add_argument('--path_image', default='./twitter_subimages', help='path to images')
    parser.add_argument('--crop_size', type=int, default = 224, help='crop size of image')
    parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained CNN if True')
    parser.add_argument('--att_mode', choices=['text', 'vis_only', 'vis_concat',  'vis_att', 'vis_concat_attimg', \
                                                'text_vis_att_img_gate', 'vis_att_concat', 'vis_att_attimg', \
    'vis_att_img_gate', 'vis_concat_attimg_gate'], default ='vis_concat_attimg_gate', \
    help='different attention mechanism')
    parser.add_argument('--resnet_root', default='./resnet', help='path the pre-trained cnn models')
    parser.add_argument('--checkpoint', default='./checkpoint/', help='path to checkpoint prefix')
    parser.add_argument('--load_check_point', action='store_true', help='path of checkpoint')
    parser.add_argument('--load_opt', action='store_true', help='load optimizer from ')
    parser.add_argument('--tfn', action='store_true', help='whether to use TFN')

    return parser.parse_args()

def configure_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    return device, n_gpu
  
def configure_dataset(opt):
    if opt.dataset == "mvsa-m-100":
        opt.path_image = "../../Datasets/MVSA-m/ESAFN-modified/mvsa-m-100/images"
        opt.max_seq_len = 24
        opt.rand_seed = 25
    else:
        print("The dataset name is incorrect!")
    
    return opt

def initialize_model_components(opt):
    model_classes = {
        # 'mmian': MMIAN,
        # 'mmram': MMRAM,
        # 'mmmgan': MMMGAN,
        'mmfusion': MMFUSION
    }
    input_colses = {
        # 'mmian': ['text_raw_without_aspect_indices', 'aspect_indices'],
        # 'mmram': ['text_raw_indices', 'aspect_indices'],
        # 'mmmgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        # 'mmfusion': ['text_left_indicator', 'text_right_indicator', 'aspect_indices']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }
    
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    
    return opt

def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(true, preds, average='macro', zero_division=0)
    #f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro
    
def save_checkpoint(state, track_list, filename):
    with open(filename+'.json', 'w') as f:
        json.dump(track_list, f)
    torch.save(state, filename+'.model')

def print_gpu_memory():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached:    {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
class Instructor:
    def __init__(self, opt):
        self.opt = opt
        
        print('> training arguments:')
        for arg in vars(opt):
            print('>>> {0}: {1}'.format(arg, getattr(opt, arg)))
            
        if not os.path.exists(opt.checkpoint):
            os.mkdir(opt.checkpoint)

def main():
    
    # Initialize hyperperameters
    opt = parse_arguments()
    opt = configure_dataset(opt)
    opt = initialize_model_components(opt)
    device, n_gpu = configure_device()
    opt.device = device
    
    if opt.tfn:
        print("************add another tfn layer*************")
    else:
        print("************no tfn layer************")
    
    # Set random seeds
    random.seed(opt.rand_seed)
    np.random.seed(opt.rand_seed)
    torch.manual_seed(opt.rand_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(opt.rand_seed)
    
    # Instantiate and run the instructor
    ins = Instructor(opt)
    ins.run()

if __name__ == "__main__":
    main()
