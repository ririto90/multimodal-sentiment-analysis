# data_utils.py

import os
import sys
import time
import pickle
import numpy as np
import re
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

from Project.settings import DATASET_PATHS, IMAGE_PATH, GLOVE_BASE_PATH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download NLTK
nltk.data.path.append('/usr/local/Caskroom/miniconda/base/envs/ml/nltk_data')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

def preprocess_tweet(tweet):
    tweet = re.sub(r'#\w+', '', tweet)  # Remove hashtags
    tweet = re.sub(r'http\S+', '', tweet)  # Remove URLs
    tokens = word_tokenize(tweet.lower())
    tokens = [word for word in tokens if word.isalpha()]
    return tokens

def load_word_vec(path, word2idx=None):
    word_vec = {}
    with open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        for line in fin:
            tokens = line.rstrip().split(' ')
            if word2idx is None or tokens[0] in word2idx:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec

def build_embedding_matrix(word2idx, embed_dim, type, glove=GLOVE_BASE_PATH):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.dat'.format(str(embed_dim), type)
    print('loading word vectors...')
    embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))
    fname = glove + 'glove.twitter.27B.' + str(embed_dim) + 'd.txt'
    word_vec = load_word_vec(fname, word2idx=word2idx)
    print('building embedding_matrix:', embedding_matrix_file_name)
    for word, i in word2idx.items():
        vec = word_vec.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
    with open(embedding_matrix_file_name, 'wb') as f:
        pickle.dump(embedding_matrix, f)
    return embedding_matrix

class Tokenizer(object):
    def __init__(self, lower=False, max_seq_len=None):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {'<PAD>': 1}
        self.idx2word = {1: '<PAD>'}
        self.idx = 2

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    @staticmethod
    def pad_sequence(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
        x = (np.ones(maxlen) * value).astype(dtype)
        trunc = sequence[:maxlen] if truncating == 'post' else sequence[-maxlen:]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def text_to_sequence(self, text, reverse=False):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx.get(w, unknownidx) for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return Tokenizer.pad_sequence(sequence, self.max_seq_len, padding='post', truncating='post')

class MVSADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class MVSADatasetReader:
    @staticmethod
    def __read_text__(fnames):
        all_tokens = []
        for fname in fnames:
            with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
                lines = fin.readlines()
                for i in range(1, len(lines)):
                    raw_text = lines[i].split('\t')[2].strip().lower()
                    tokens = preprocess_tweet(raw_text)
                    all_tokens.extend(tokens)
        return ' '.join(all_tokens)

    @staticmethod
    def __read_data__(fname, tokenizer, max_seq_len):
        start_time = time.time()
        print(f'-------------- Loading {fname} ---------------')
        data = []
        num_classes = set()
        seq_lengths = []
        
        with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            lines = fin.readlines()
            for i in range(1, len(lines)):
                line_parts = lines[i].split('\t')
                idx = int(line_parts[0].strip())
                sentiment = int(line_parts[1].strip())
                raw_text = line_parts[2].strip()  # you can .lower() in preprocess_tweet if you like
                tokens = preprocess_tweet(raw_text)
                seq_lengths.append(len(tokens))
                processed_str = ' '.join(tokens)
                num_classes.add(sentiment)
                
                text_indices = tokenizer.text_to_sequence(processed_str)
                sample = {
                    'raw_text': raw_text,
                    'text_indices': torch.tensor(text_indices, dtype=torch.long),
                    'polarity': sentiment,
                }
                data.append(sample)
                if i < 5:  # Print only the first 5 lines for brevity
                    print(f"[DEBUG] index: {idx}")
                    print(f"[DEBUG] raw_text: {raw_text}")
                    print(f"[DEBUG] processed_str: {processed_str}")
                    print(f"[DEBUG] text_indices: {text_indices}")
                    print(f"[DEBUG] polarity: {sentiment}")
                    
        end_time = time.time()
        print(f"Time taken to load {fname}: {end_time - start_time:.2f} seconds"
              f"({(end_time - start_time) / 60:.2f} minutes)")
        return data, num_classes, seq_lengths

    def __init__(self, paths=DATASET_PATHS, dataset='mvsa-mts', max_seq_len=40):
        
        self.all_seq_lengths = []
        
        # Dataset Paths
        dataset_paths = DATASET_PATHS.get(dataset)
        if dataset_paths is None:
            raise ValueError(f"Dataset {dataset} not found in DATASET_PATHS")
        train_path = dataset_paths["train"]
        val_path   = dataset_paths["val"]
        test_path  = dataset_paths["test"]
        
        print(f"Loading dataset '{dataset}':\n  Train path: {train_path}\n"
              f"\tValidation path: {val_path}\n  Test path: {test_path}")
        
        # Tokenizer
        text = MVSADatasetReader.__read_text__([train_path, val_path, test_path])
        tokenizer = Tokenizer(max_seq_len=max_seq_len)
        tokenizer.fit_on_text(text.lower())
        self.tokenizer = tokenizer
        
        # Embedding Matrix
        embed_dim = 200
        embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim=embed_dim, type='glove')
        self.embedding_matrix = embedding_matrix
        
        # Load Dataset Split (Train)
        train_data, train_classes, train_lengths = MVSADatasetReader.__read_data__(train_path, tokenizer, max_seq_len)
        self.all_seq_lengths.extend(train_lengths)
        print(f"Train classes: {sorted(list(train_classes))}, count={len(train_classes)}")
            
        print("[DEBUG] Train label distribution:")
        labels_train = [item['polarity'] for item in train_data]
        unique, counts = np.unique(labels_train, return_counts=True)
        print(dict(zip(unique, counts)))
        
        # Dynamically compute weights from training distribution
        total_train = sum(counts)
        num_classes = len(unique)  # e.g., 3

        # Sort `unique` to ensure label order is [0,1,2,...]
        # Then compute weight for each label in that order
        class_count_map = dict(zip(unique, counts)) 
        weights = []
        for label in range(num_classes):
            count = class_count_map[label]
            # One common formula: total / (num_classes * count)
            w = total_train / (num_classes * count)
            weights.append(w)
        
        # Store as a torch tensor so we can directly use it in CrossEntropyLoss
        self.class_weights = torch.tensor(weights, dtype=torch.float)

        # Load Dataset Split (Val)
        val_data, val_classes, val_lengths = MVSADatasetReader.__read_data__(val_path, tokenizer, max_seq_len)
        self.all_seq_lengths.extend(val_lengths)
        print(f"Val classes: {sorted(list(val_classes))}, count={len(val_classes)}")
            
        print("[DEBUG] Train label distribution:")
        labels_train = [item['polarity'] for item in val_data]
        unique, counts = np.unique(labels_train, return_counts=True)
        print(dict(zip(unique, counts)))
        
        # Print for debugging
        print(f"[DEBUG] Computed class_weights = {self.class_weights.tolist()}")
        
        # Load Dataset Split (Test)
        test_data, test_classes, test_lengths = MVSADatasetReader.__read_data__(test_path, tokenizer, max_seq_len)
        self.all_seq_lengths.extend(test_lengths)
        print(f"Test classes: {sorted(list(test_classes))}, count={len(test_classes)}")
        
        print("[DEBUG] Train label distribution:")
        labels_train = [item['polarity'] for item in test_data]
        unique, counts = np.unique(labels_train, return_counts=True)
        print(dict(zip(unique, counts)))
        
        optimal_seq_len = int(np.percentile(self.all_seq_lengths, 95))
        print(f"[DEBUG] 95th percentile sequence length across all splits: {optimal_seq_len:.2f}")
        
        # Dataset Loaders
        self.train_data = MVSADataset(train_data)
        self.val_data = MVSADataset(val_data)
        self.test_data = MVSADataset(test_data)
        
        # Unique Classes & Total Training Samples
        all_unique_classes = train_classes.union(val_classes).union(test_classes)
        self.num_classes = len(all_unique_classes)
        total_training_samples = len(train_data) + len(val_data) + len(test_data)
        print(f'Total Training Samples: {total_training_samples}')
        print(f'Number of Training Samples: {len(train_data)}')
        print(f'Number of Validation Samples: {len(val_data)}')
        print(f'Number of Test Samples: {len(test_data)}')
        print(f"Number of unique sentiment classes: {self.num_classes}")
