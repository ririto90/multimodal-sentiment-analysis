# train.py

import os
import sys
import argparse
import random
import numpy as np
import time
import torch

from instructor import Instructor
from models.dmlanfusion import DMLANFUSION

print("Python PATH:", sys.path)
log_dir = os.getenv('NEW_LOGS_DIR')
if log_dir is None:
    raise ValueError("NEW_LOGS_DIR environment variable is not set")
print(f"Logs directory: {log_dir}")

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rand_seed', default=8, type=int)
    parser.add_argument('--model_fusion', default='mmfusion', type=str)
    parser.add_argument('--dataset', default='mvsa-mts-100', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--num_epoch', default=8, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--max_seq_len', default=64, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    parser.add_argument('--path_image', default='./images')
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--n_head', type=int, default=8)
    
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=3)
    
    parser.add_argument('--log_dir', default=log_dir, type=str)
    
    parser.add_argument('--counter', default=0, type=int)

    opt = parser.parse_args()

    random.seed(opt.rand_seed)
    np.random.seed(opt.rand_seed)
    torch.manual_seed(opt.rand_seed)

    model_classes = {
        'dmlanfusion': DMLANFUSION
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

    opt.model_class = model_classes[opt.model_fusion]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    ins = Instructor(opt)
    ins.run()
    time.sleep(10)
    
    ins.read_tensorboard_loss(log_dir)
    ins.plot_tensorboard_loss(log_dir=log_dir)
    
    end_time = time.time()
    print(f"Total Completion Time: {(end_time - start_time) / 60:.2f} minutes. ({(end_time - start_time) / 3600:.2f} hours) ")
