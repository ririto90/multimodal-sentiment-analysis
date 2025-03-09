# data_utils.py

import os
import time
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer
from helpers import *

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
    
def overall_sentiment(row):
    print("DEBUG row:", row['text'], row['image'])  # which numeric values?
    if (row['text'] == 0 and row['image'] == 2) or (row['text'] == 2 and row['image'] == 0):
        print(" -> negative")
        return 0
    elif (row['text'] == 1 and row['image'] == 2) or (row['text'] == 2 and row['image'] == 1):
        print(" -> positive")
        return 1
    print(" -> same as text:", row['text'])
    return row['text']

class MOAMVSADataSet:
    def __init__(self, current_working_directory, dataset_name):
        self.dataset_name = dataset_name # 'MVSA-single' 'MVSA-multiple'
        self.dataset_dir = os.path.join(current_working_directory, 'Datasets', self.dataset_name)
        self.cwd = current_working_directory
        self.texts_path  = os.path.join(self.dataset_dir, 'text')
        self.images_path = os.path.join(self.dataset_dir, 'image')

        os.chdir(self.dataset_dir)

        labels = pd.read_csv('labels.csv', sep='\t').dropna()
        labels = labels.sort_values(by = ['ID'])
        labels = labels.replace(sentiment_label)
        print(labels.head(10))
        labels['overall_sentiment'] = labels.apply(lambda row: overall_sentiment(row), axis=1)

        # print('Distribuția claselor în %{self.dataset_name}:')
        # print('Text:\n', labels['text'].value_counts())
        # print('Images:\n', labels['image'].value_counts())
        # print('Posts:\n', labels['overall_sentiment'].value_counts())
        
        # load raw data into variable from the text and image files
        os.chdir(self.texts_path)
        text_filenames = [file for file in glob.glob("*.txt")]

        os.chdir(self.images_path)
        image_files = [file for file in glob.glob("*.jpg")]

        # Split the text set in train, validation and test sets with 8:1:1 ratio
        text_files_train, text_files_test, image_files_train, image_files_test = \
            train_test_split(text_filenames, image_files, test_size = 0.1)
        text_files_train, text_files_val, image_files_train, image_files_val = \
            train_test_split(text_files_train, image_files_train, test_size = 0.1)

        text_files_train, image_files_train = sorted(text_files_train, key = get_file_index), sorted(image_files_train, key = get_file_index)
        text_files_val, image_files_val = sorted(text_files_val, key = get_file_index), sorted(image_files_val, key = get_file_index)
        text_files_test, image_files_test = sorted(text_files_test, key = get_file_index), sorted(image_files_test, key = get_file_index)

        idx_train = list(map(get_file_index, text_files_train))
        idx_val = list(map(get_file_index, text_files_val))
        idx_test = list(map(get_file_index, text_files_test))

        df_text_train = get_text_dataframe(self.texts_path, text_files_train)
        df_text_val = get_text_dataframe(self.texts_path, text_files_val)
        df_text_test = get_text_dataframe(self.texts_path, text_files_test)

        """
            Pre-processing the text data
        """
        # STEP 1: Convert emojis to their corresponding words
        # df_text_train['text'] = df_text_train['text'].apply(demojize)
        # df_text_val['text'] = df_text_val['text'].apply(demojize)
        # df_text_test['text'] = df_text_test['text'].apply(demojize)
        # STEP 2: Convert emoticons to their corresponding words
        # df_text_train['text'] = df_text_train['text'].apply(replace_emoticons)
        # df_text_val['text'] = df_text_val['text'].apply(replace_emoticons)
        # df_text_test['text'] = df_text_test['text'].apply(replace_emoticons)
        # STEP 3: Decode text in "UTF-8" and normalize it
        df_text_train['text'] = df_text_train['text'].apply(lambda x : normalize_text(x))
        df_text_val['text'] = df_text_val['text'].apply(lambda x : normalize_text(x))
        df_text_test['text'] = df_text_test['text'].apply(lambda x : normalize_text(x))
        # STEP 4: Replace URLs with the <URL> tag
        df_text_train['text'] = df_text_train['text'].apply(lambda x : re.sub(url_addresses_reg, ' <URL> ', x))
        df_text_val['text'] = df_text_val['text'].apply(lambda x : re.sub(url_addresses_reg, ' <URL> ', x))
        df_text_test['text'] = df_text_test['text'].apply(lambda x : re.sub(url_addresses_reg, ' <URL> ', x))
        # STEP 5: Remove all the email addresses
        df_text_train['text'] = df_text_train['text'].apply(lambda x : re.sub(mail_reg, ' ', x))
        df_text_val['text'] = df_text_val['text'].apply(lambda x : re.sub(mail_reg, ' ', x))
        df_text_test['text'] = df_text_test['text'].apply(lambda x : re.sub(mail_reg, ' ', x))
        # STEP 6: Split attached words
        df_text_train['text'] = df_text_train['text'].apply(split_attached_words)
        df_text_val['text'] = df_text_val['text'].apply(split_attached_words)
        df_text_test['text'] = df_text_test['text'].apply(split_attached_words)
        # STEP 7: Lower casing all tweets
        df_text_train['text'] = df_text_train['text'].apply(lambda x : x.lower())
        df_text_val['text'] = df_text_val['text'].apply(lambda x : x.lower())
        df_text_test['text'] = df_text_test['text'].apply(lambda x : x.lower())
        # STEP 8: Replace any sequence of the same letter of length greater than 2 with a sequence of length 2
        df_text_train['text'] = df_text_train['text'].apply(remove_multiple_occurences)
        df_text_val['text'] = df_text_val['text'].apply(remove_multiple_occurences)
        df_text_test['text'] = df_text_test['text'].apply(remove_multiple_occurences)
        # STEP 9: Abbreviated words are extended to the form in the dictionary
        df_text_train['text'] = df_text_train['text'].apply(expand_words)
        df_text_val['text'] = df_text_val['text'].apply(expand_words)
        df_text_test['text'] = df_text_test['text'].apply(expand_words)
        # STEP 10: Eliminate numerical and special characters 
        df_text_train['text'] = df_text_train['text'].apply(lambda x : x.translate(str.maketrans(' ', ' ', special_characters)))
        df_text_val['text'] = df_text_val['text'].apply(lambda x : x.translate(str.maketrans(' ', ' ', special_characters)))
        df_text_test['text'] = df_text_test['text'].apply(lambda x : x.translate(str.maketrans(' ', ' ', special_characters)))
        # # STEP 11(Optional): Remove the old style retweet text
        df_text_train['text'] = df_text_train['text'].apply(lambda x : re.sub(retweets_reg, ' ', x))
        df_text_val['text'] = df_text_val['text'].apply(lambda x : re.sub(retweets_reg, ' ', x))
        df_text_test['text'] = df_text_test['text'].apply(lambda x : re.sub(retweets_reg, ' ', x))

        # Apply the padding strategy
        train_text_lengths = df_text_train['text'].apply(lambda x : len(x.split(' ')))
        val_text_lengths = df_text_val['text'].apply(lambda x : len(x.split(' ')))
        test_text_lengths = df_text_test['text'].apply(lambda x : len(x.split(' ')))

        self.max_text_length = int(max(np.mean(train_text_lengths), np.mean(val_text_lengths), \
            np.mean(test_text_lengths))) + 5

        images_train = get_images(self.images_path, image_files_train)
        images_val = get_images(self.images_path, image_files_val)
        images_test = get_images(self.images_path, image_files_test)

        y_train = labels[labels['ID'].isin(idx_train)]
        y_train = y_train.sort_values(by = ['ID'])
        del(idx_train)

        y_val = labels[labels['ID'].isin(idx_val)]
        y_val = y_val.sort_values(by = ['ID'])
        del(idx_val)

        y_test = labels[labels['ID'].isin(idx_test)]
        y_test = y_test.sort_values(by = ['ID'])
        del(idx_test)


        self.__X_text_train = df_text_train['text'].values
        self.__X_text_val = df_text_val['text'].values
        self.__X_text_test = df_text_test['text'].values
        self.__y_text_train = y_train['text'].values
        self.__y_text_val = y_val['text'].values
        self.__y_text_test = y_test['text'].values
        self.__X_images_train = images_train
        self.__X_images_val = images_val
        self.__X_images_test = images_test
        self.__y_images_train = y_train['image'].values
        self.__y_images_val = y_val['image'].values
        self.__y_images_test = y_test['image'].values

        # self.__y_post_train = self.get_labels(y_train['text'].values, y_train['image'].values)
        # self.__y_post_val = self.get_labels(y_val['text'].values, y_val['image'].values)
        # self.__y_post_test = self.get_labels(y_test['text'].values, y_test['image'].values)
        self.__y_post_train = y_train['overall_sentiment'].values
        self.__y_post_val = y_val['overall_sentiment'].values
        self.__y_post_test = y_test['overall_sentiment'].values
        
        print("\n[DEBUG] First 5 training texts:")
        for i in range(min(5, len(df_text_train))):
            print(
                f"    ID: {y_train['ID'].iloc[i]} | "
                f"Overall sentiment: {y_train['overall_sentiment'].iloc[i]} | "
                f"Text: {df_text_train['text'].iloc[i]}"
            )

    def get_labels(self, y_text, y_images):
        y_text, y_images = np.array(y_text), np.array(y_images)
        y_post = [get_label(a, b) for a, b in zip(y_text, y_images)]
        y_post = np.array(y_post).astype('float32')

        return y_post

    @property
    def text_train(self):
        return np.array(self.__X_text_train, dtype = str)

    @property
    def text_validation(self):
        return np.array(self.__X_text_val, dtype = str)

    @property
    def text_test(self):
        return np.array(self.__X_text_test, dtype = str)

    @property
    def images_train(self):
        return self.__X_images_train

    @property
    def images_validation(self):
        return self.__X_images_val

    @property
    def images_test(self):
        return self.__X_images_test

    @property
    def text_train_labels(self):
        return np.array(self.__y_text_train).astype('float32')

    @property
    def text_validation_labels(self):
        return np.array(self.__y_text_val).astype('float32')

    @property
    def text_test_labels(self):
        return np.array(self.__y_text_test).astype('float32')

    @property
    def image_train_labels(self):
        return np.array(self.__y_images_train).astype('float32')

    @property
    def image_validation_labels(self):
        return np.array(self.__y_images_val).astype('float32')

    @property
    def image_test_labels(self):
        return np.array(self.__y_images_test).astype('float32')

    @property
    def post_train_labels(self):
        return self.__y_post_train

    @property
    def post_val_labels(self):
        return self.__y_post_val

    @property
    def post_test_labels(self):
        return self.__y_post_test


class TextDataset(Dataset):
    def __init__(self, X, y, max_length, device):
        self.X = X
        self.y = y
        self.max_length = max_length
        self.device = device

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_punctuation_indexes(self, tokens):
        indexes = []
        for idx, token in enumerate(tokens[:-1]):
            if token in punctuation and not tokens[idx + 1] in punctuation:
                indexes.append(idx)

        return indexes

    def add_special_tokens(self, tokens, punctuation_indexes):
        for i, index in enumerate(punctuation_indexes):
            tokens = tokens[:index + i + 1] + ['[SEP]'] + tokens[index + i + 1:]

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        return tokens

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]

        # Pre-processing the data to be suitable for the BERT model
        tokens = self.tokenizer.tokenize(sample)
        indexes = self.get_punctuation_indexes(tokens)
        tokens = self.add_special_tokens(tokens, indexes)

        if len(tokens) < self.max_length:
            tokens = tokens + ['[PAD]' for _ in range(self.max_length - len(tokens))]
        else:
            tokens = tokens[:self.max_length-1] + ['[SEP]'] # Prunning the list to be of specified max length

        # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
        tokens_ids = torch.tensor(tokens_ids, device=self.device) 

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids != 0).long()
        attn_mask = torch.tensor(attn_mask, device=self.device)

        label = torch.tensor(label, device=self.device)

        return tokens_ids, attn_mask, label


class MultimodalDataset(Dataset):
    def __init__(self, X_image, X_text, y, max_length):
        self.X = X_text
        self.X_image = X_image
        self.y = y
        self.max_length = max_length

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_punctuation_indexes(self, tokens):
        indexes = []
        for idx, token in enumerate(tokens[:-1]):
            if token in punctuation and not tokens[idx + 1] in punctuation:
                indexes.append(idx)

        return indexes

    def add_special_tokens(self, tokens, punctuation_indexes):
        for i, index in enumerate(punctuation_indexes):
            tokens = tokens[:index + i + 1] + ['[SEP]'] + tokens[index + i + 1:]

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        return tokens

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        image = self.X_image[idx]
        label = self.y[idx]

        # Pre-processing the data to be suitable for the BERT model
        tokens = self.tokenizer.tokenize(sample)
        indexes = self.get_punctuation_indexes(tokens)
        tokens = self.add_special_tokens(tokens, indexes)

        if len(tokens) < self.max_length:
            tokens = tokens + ['[PAD]' for _ in range(self.max_length - len(tokens))]
        else:
            tokens = tokens[:self.max_length-1] + ['[SEP]'] # Prunning the list to be of specified max length

        # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
        tokens_ids = torch.tensor(tokens_ids) 

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids != 0).long()
        attn_mask = torch.tensor(attn_mask)

        label = torch.tensor(label)

        return tokens_ids, attn_mask, image, label
        