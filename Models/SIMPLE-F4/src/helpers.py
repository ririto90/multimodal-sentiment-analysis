# helpers.py

import os
import sys
import glob, shutil
import pickle
import argparse
from copy import deepcopy

import re
import string
import json
import unicodedata

import cv2 as cv

import random
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models
from PIL import Image

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer, BertModel, AdamW

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

base_dir = '/home/rgg2706/Multimodal-Sentiment-Analysis/Models'
additional_files_dir = os.path.join(base_dir, 'MultimodalOpinionAnalysis2/AdditionalFiles')
datasets_names = ['MVSA-single', 'MVSA-multiple']

punctuation = string.punctuation
sentiment_label = {"negative": 0, "positive": 1, "neutral": 2}

# STOPWORDS = set(stopwords.words("english"))
special_characters = punctuation.replace("!", "").replace(".", "").replace("?", "") + string.digits

mail_reg = r'^(\w|\.|\_|\-)+[@](\w|\_|\-|\.)+[.]\w{2,3}$'
url_addresses_reg = r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?|http:/\W"|http:\/\/\w\W\.\W'
retweets_reg = r'^rt[\s]+|rt '

with open(additional_files_dir + "/contractions.json") as f:
    contractions = json.loads(f.read())


def get_file_index(file):
    try:
        file_name, extension = file.split('.')
    except ValueError:
        return None
    
    return int(file_name)


def join_path(*paths):
    return os.path.sep.join(os.path.normpath(path.rstrip("/")) for path in paths)


def move_files(files_names, extension, files_path, destination_path):
    # change the working directory to the directory where we have all the date files of the current dataset
    os.chdir(files_path)
    for file in files_names:
        shutil.copy(files_path + '/' + file + extension, destination_path)


def rearrange_text_dataframe(df):
    new_df = pd.concat(df)
    new_df = new_df.reset_index(drop=True) # resetting the index and removing old index
    new_df['text'] = new_df[new_df.columns[:]].apply(lambda x : ' '.join(x.dropna().astype(str)), axis=1)
    new_df = new_df[['text']].astype(str)

    return new_df


def read_line(input_file_path):
    input_file = open(input_file_path, 'r')

    while True:
        line = input_file.readline()
        yield line


def read_image(path):
    try:
        img = cv.imread(path)
        if img is not None:
            img = cv.resize(img, (128, 128))
            img = img.astype(np.float32) / 255.
        else:
            print('Read failed:', path)
    except:
        print('Unexpected error: ', sys.exc_info()[0])
        print(sys.exc_info()[1])
        sys.exit(1)

    return img


def get_images(path, images_filenames):
    # Path to your fallback image
    fallback_path = "/home/rgg2706/Multimodal-Sentiment-Analysis/Datasets/0default.jpg"

    images = []
    for image_filename in images_filenames:
        full_path = os.path.join(path, image_filename)
        try:
            input_image = Image.open(full_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            # Use the fallback image
            try:
                input_image = Image.open(fallback_path).convert('RGB')
                print(f"  [*] Using fallback image: {fallback_path}")
            except Exception as e_fallback:
                # If the fallback also fails, skip or raise
                print(f"  [!] Fallback also failed: {fallback_path} => {e_fallback}")
                continue  # or raise, or append some default tensor

        # Preprocessing
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        images.append(input_tensor)

    return images


def read_textfile(path):
    lines = []
    with open(path, errors='ignore') as f:
        lines = f.readlines()
    
    lines = ' '.join(lines)
    return lines

def get_text_dataframe(path, text_filenames):
    df_text = [read_textfile(join_path(path, file)) for file in text_filenames]
    df_text = pd.DataFrame(df_text, columns = ['text'])
    return df_text


def normalize_text(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore') # remove accented chars
    return x


def expand_words(tweet):
    for key, value in contractions.items():
        tweet = tweet.replace(key, value)
    return tweet


def replace_char(old_str, pos, chr):
    list_chrs = list(old_str)
    list_chrs[pos] = chr
    return ''.join(list_chrs)


"""
    any letter repeated more than three times in a row is replaced by two repetitions of the same letter
"""
def remove_multiple_occurences(text):
    n = len(text)

    if n < 3:
        return text

    i, count = 0, 0
    while i < n - 1:
        i += 1
        if text[i] != text[i-1]:
            count = 0
        else:
            count += 1
            if count >= 2:
                text = text[:i] + text[i+1:]
                n -= 1
                i -= 1

    return text


def get_label(a, b):
    if (a == 0 and b == 2) or (a == 2 and b == 0):
        return 0
    elif (a == 1 and b == 2) or (a == 2 and b == 1):
        return 1
    else:
        return a


def split_attached_words(text):
    text = " ".join([groupe for groupe in re.split(r"([A-Z][a-z]+[^A-Z]*)", text) if groupe])
    return text


def get_recall(tp, fn):        
  recall_negative = tp[0] / (tp[0] + fn[0] + 1e-5)
  recall_positive = tp[1] / (tp[1] + fn[1] + 1e-5)
  recall_neutral = tp[2] / (tp[2] + fn[2] + 1e-5)

  return [recall_negative, recall_neutral, recall_positive]


def get_avg_recall(recall):
  return np.sum(recall) / 3.0


def get_accuracy_from_logits(logits, labels):
  logs = logits.detach().to("cpu").numpy()
  logs = logs.argmax(axis=1)
  labels = labels.detach().to("cpu").numpy()
  accuracy = np.sum(logs == labels) / len(labels)
  
  return accuracy


def get_tp_fn_fp_from_logits(logits, labels):
  tp, fn , fp = [0, 0, 0], [0, 0, 0], [0, 0, 0]

  logs = logits.detach().to("cpu").numpy()
  labels = labels.detach().to("cpu").numpy()
  logs = logs.argmax(axis=1)
  try:
    logs = np.squeeze(logs)
  except ValueError:
    print('Can\'t squeeze shape: ', logs.shape)

  for (predicted, actual) in zip(logs, labels):
    actual = int(actual)
    if predicted == actual:
        tp[actual] += 1
    else:
        fp[predicted] += 1
        fn[actual] += 1

  return tp, fn, fp


def get_f1_pn(tp, fp, fn):
  recall_negative = tp[0] / (tp[0] + fn[0] + 1e-5)
  recall_positive = tp[1] / (tp[1] + fn[1] + 1e-5)

  precision_negative = tp[0] / (tp[0] + fp[0] + 1e-5)
  precision_positive = tp[1] / (tp[1] + fp[1] + 1e-5)

  f1_negative = 2. * (precision_negative * recall_negative) / (precision_negative + recall_negative + 1e-5)
  f1_positive = 2. * (precision_positive * recall_positive) / (precision_positive + recall_positive + 1e-5)

  f1_pn = .5 * (f1_negative + f1_positive)
  return f1_pn


def plot_losses(train_loss, val_loss, test_loss):
    pyplot.plot(train_loss, label = "Train Loss", color = "green")
    pyplot.plot(val_loss, label = "Validation Loss", color = "blue")
    pyplot.plot(test_loss, label = "Test Loss", color = "red")
    pyplot.legend()
    pyplot.show()
    