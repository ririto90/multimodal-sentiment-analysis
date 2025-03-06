# data_utils.py

import os
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def image_process(image_path, transform):
    img = Image.open(image_path).convert('RGB')
    return transform(img)

class MVSADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class MVSADatasetReader:
    def __init__(self, paths, path_image, dataset='mvsa-mts', max_seq_len=40):
        """
        :param paths:        e.g. DATASET_PATHS, a dict with subdict for each dataset
        :param path_image:   Directory with .jpg images
        :param dataset:      The dataset key in `paths`
        :param max_seq_len:  Maximum BERT tokenizer sequence length
        :param pretrained_model_name: e.g. 'bert-base-uncased'
        """
        self.all_seq_lengths = []

        # 1) Resolve dataset paths
        dataset_paths = paths.get(dataset)
        if dataset_paths is None:
            raise ValueError(f"Dataset {dataset} not found in the provided paths dictionary.")
        train_path = dataset_paths["train"]
        val_path   = dataset_paths["val"]
        test_path  = dataset_paths["test"]

        print(f"Loading dataset '{dataset}':")
        print(f"  Train path: {train_path}")
        print(f"  Validation path: {val_path}")
        print(f"  Test path: {test_path}")

        # 2) BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_len = max_seq_len
        self.path_image = path_image

        # 3) Load train, val, test splits
        train_data, train_classes, train_lengths = self._read_data(train_path)
        self.all_seq_lengths.extend(train_lengths)
        print(f"Train classes: {sorted(list(train_classes))}, count={len(train_classes)}")

        print("[DEBUG] Train label distribution:")
        labels_train = [item['polarity'] for item in train_data]
        unique, counts = np.unique(labels_train, return_counts=True)
        print(dict(zip(unique, counts)))

        # Dynamically compute weights from the training distribution
        total_train = sum(counts)
        num_classes = len(unique)
        class_count_map = dict(zip(unique, counts))
        weights = []
        # Ensure a sorted iteration of unique labels
        for label in sorted(class_count_map.keys()):
            count = class_count_map[label]
            w = total_train / (num_classes * count)  # e.g. total / (num_classes*count)
            weights.append(w)
        self.class_weights = torch.tensor(weights, dtype=torch.float)

        # Validation split
        val_data, val_classes, val_lengths = self._read_data(val_path)
        self.all_seq_lengths.extend(val_lengths)
        print(f"Val classes: {sorted(list(val_classes))}, count={len(val_classes)}")

        print("[DEBUG] Val label distribution:")
        labels_val = [item['polarity'] for item in val_data]
        unique_val, counts_val = np.unique(labels_val, return_counts=True)
        print(dict(zip(unique_val, counts_val)))

        print(f"[DEBUG] Computed class_weights = {self.class_weights.tolist()}")

        # Test split
        test_data, test_classes, test_lengths = self._read_data(test_path)
        self.all_seq_lengths.extend(test_lengths)
        print(f"Test classes: {sorted(list(test_classes))}, count={len(test_classes)}")

        print("[DEBUG] Test label distribution:")
        labels_test = [item['polarity'] for item in test_data]
        unique_test, counts_test = np.unique(labels_test, return_counts=True)
        print(dict(zip(unique_test, counts_test)))

        # 95th percentile sequence length across all splits
        if self.all_seq_lengths:
            optimal_seq_len = int(np.percentile(self.all_seq_lengths, 95))
        else:
            optimal_seq_len = 0
        print(f"[DEBUG] 95th percentile sequence length across all splits: {optimal_seq_len:.2f}")

        # 4) Wrap into Dataset objects
        self.train_data = MVSADataset(train_data)
        self.val_data   = MVSADataset(val_data)
        self.test_data  = MVSADataset(test_data)

        # 5) Summaries
        all_unique_classes = train_classes.union(val_classes).union(test_classes)
        self.num_classes = len(all_unique_classes)
        total_samples = len(train_data) + len(val_data) + len(test_data)

        print(f"Total Training Samples: {total_samples}")
        print(f"Number of Training Samples: {len(train_data)}")
        print(f"Number of Validation Samples: {len(val_data)}")
        print(f"Number of Test Samples: {len(test_data)}")
        print(f"Number of unique sentiment classes: {self.num_classes}")

    def _read_data(self, fname):
        start_time = time.time()
        print(f'-------------- Loading {fname} ---------------')

        data = []
        num_classes = set()
        seq_lengths = []

        with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            lines = fin.readlines()
            for i in range(1, len(lines)):
                line_parts = lines[i].split('\t')
                if len(line_parts) < 3:
                    continue

                idx_str   = line_parts[0].strip()
                sentiment = int(line_parts[1].strip())
                raw_text  = line_parts[2].strip()

                # For debug, track approximate text length
                seq_len = len(raw_text.split())
                seq_lengths.append(seq_len)
                num_classes.add(sentiment)

                # BERT Tokenization
                encoding = self.tokenizer(
                    raw_text,
                    max_length=self.max_seq_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                # Image loading
                image_path = os.path.join(self.path_image, idx_str + ".jpg")
                try:
                    img_tensor = image_process(image_path, image_transform)
                except Exception:
                    # If fail, fallback
                    fallback = os.path.join(self.path_image, '0default.jpg')
                    img_tensor = image_process(fallback, image_transform)

                sample = {
                    'raw_text': raw_text,
                    'input_ids': encoding['input_ids'].squeeze(0),    # [max_seq_len]
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'images': img_tensor,                             # [3, 224, 224]
                    'polarity': sentiment
                }
                data.append(sample)

                # Debug for first 5 lines
                if i <= 5:
                    print(f"[DEBUG] index: {idx_str}")
                    print(f"[DEBUG] raw_text: {raw_text}")
                    print(f"[DEBUG] text_length: {seq_len}")
                    print(f"[DEBUG] polarity: {sentiment}")
                    print(f"[DEBUG] first 10 input_ids: {sample['input_ids'][:10].tolist()}")
                    print("---")

        end_time = time.time()
        print(f"Time taken to load {fname}: {end_time - start_time:.2f} seconds "
              f"({(end_time - start_time) / 60:.2f} minutes)")

        return data, num_classes, seq_lengths