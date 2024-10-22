# grid_search.py

import itertools
import os
import copy
import random
import numpy as np
import torch
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
  
    # Base Hyperparameters
    base_opt = Namespace(
        rand_seed=8,
        model_name='mmfusion',
        dataset='mvsa-mts-100',
        optimizer='adam',
        initializer='xavier_uniform_',
        learning_rate=0.0001,
        dropout_rate=0.5,
        num_epoch=8,
        batch_size=16,
        log_step=10,
        max_seq_len=64,
        polarities_dim=3,
        clip_grad=5.0,
        path_image='./Datasets/MVSA-MTS/images-indexed',
        crop_size=224,
        n_head=8,
        common_dim=512,
        num_classes=3,
        log_dir=log_dir
    )

    # Hyperparameter grid
    hyperparameter_grid = {
        'learning_rate': [1e-4, 5e-4],
        'batch_size': [16, 32],
        'dropout_rate': [0.3, 0.5],
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

        current_opt.model_class = model_classes[current_opt.model_name]
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
