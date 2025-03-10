# grid_search.py

import itertools
import os
import copy
import random
import numpy as np
import torch
import argparse
from argparse import Namespace

from instructor import Instructor
from models.mmfusion import MMFUSION
from models.cmhafusion import CMHAFUSION
from models.mfcchfusion import MFCCHFUSION
from models.mfcchfusion2 import MFCCHFUSION2



def main():
    log_dir = os.getenv('NEW_LOGS_DIR')
    if log_dir is None:
        raise ValueError("NEW_LOGS_DIR environment variable is not set")
    print(f"Logs directory: {log_dir}")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--rand_seed', type=int, default=8)
    parser.add_argument('--model_fusion', type=str, default='mmfusion')
    parser.add_argument('--dataset', type=str, default='mvsa-mts')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--initializer', type=str, default='xavier_uniform_')
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_epoch', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--max_seq_len', type=int, default=64)
    parser.add_argument('--polarities_dim', type=int, default=3)
    parser.add_argument('--clip_grad', type=float, default=5.0)
    parser.add_argument('--path_image', type=str, default='./Datasets/MVSA-MTS/images-indexed')
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()
  
    # Base Hyperparameters
    base_opt = Namespace(
        rand_seed=args.rand_seed,
        model_fusion=args.model_fusion,
        dataset=args.dataset,
        optimizer=args.optimizer,
        initializer=args.initializer,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        log_step=args.log_step,
        max_seq_len=args.max_seq_len,
        polarities_dim=args.polarities_dim,
        clip_grad=args.clip_grad,
        path_image=args.path_image,
        crop_size=args.crop_size,
        n_head=args.n_head,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        log_dir=args.log_dir
    )

    # Hyperparameter grid
    hyperparameter_grid = {
        'learning_rate': [1e-3, 1e-4, 1e-5,
                          3e-3, 3e-4, 3e-5,
                          5e-3, 5e-4, 5e-5],
        'dropout_rate': [0.1, 0.3, 0.5],
        'hidden_dim': [512, 768, 1024]
    }

    # Generate all combinations
    keys, values = zip(*hyperparameter_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []

    for idx, params in enumerate(combinations):
        print(f"\nRunning hyperparameter combination {idx + 1}/{len(combinations)}")
        current_opt = copy.deepcopy(base_opt)
        for key, value in params.items():
            setattr(current_opt, key, value)

        # Map model classes, initializers, and optimizers
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

        current_opt.model_class = model_classes[current_opt.model_fusion]
        current_opt.initializer = initializers[current_opt.initializer]
        current_opt.optimizer = optimizers[current_opt.optimizer]

        # Set random seeds
        random.seed(current_opt.rand_seed)
        np.random.seed(current_opt.rand_seed)
        torch.manual_seed(current_opt.rand_seed)

        ins = Instructor(current_opt)
        ins.run()

        # Instance variables for performance
        dev_f1 = ins.max_dev_f1
        test_f1 = ins.max_test_f1

        results.append({
            'hyperparams': params,
            'dev_f1': dev_f1,
            'test_f1': test_f1,
        })

        print(f"Dev F1: {dev_f1:.6f}, Test F1: {test_f1:.6f}")

    # Find the best hyperparameters based on dev_f1
    best_result = max(results, key=lambda x: x['dev_f1'])
    print("\nBest Hyperparameters:")
    for key, value in best_result['hyperparams'].items():
        print(f"{key}: {value}")
    print(f"Best Dev F1: {best_result['dev_f1']:.6f}")
    print(f"Corresponding Test F1: {best_result['test_f1']:.6f}")

    output_file = os.path.join(log_dir, 'grid_search_results.json')
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
