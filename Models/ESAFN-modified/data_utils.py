# -*- coding: utf-8 -*-
# file: data_utils.py
# author: jyu5 <yujianfei1990@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def load_word_vec(path, word2idx=None):
  
	# file input (fin)
	fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
	
	word_vec = {}  # Initialize empty dic to store word vectors

	for line in fin:  # Loop each line in work vector file
	  
		# Split the line into tokens;
		# 	the first token is the word, 
		# 	and the rest are the vector values
		tokens = line.rstrip().split(' ')
		
		# If word2idx is provided, only load vectors for words that exist in word2idx
		# If word2idx is None, load vectors for all words
		if word2idx is None or tokens[0] in word2idx.keys():
			
			# Store the word and its corresponding vector in the dictionary
			word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
			
	# Return the dictionary of word vectors
	return word_vec

def build_embedding_matrix(word2idx, embed_dim, type):
	# Create a file name for the embedding matrix using the embedding dimension and type
	embedding_matrix_file_name = '{0}_{1}_embedding_matrix.dat'.format(str(embed_dim), type)
	
	load_embedding_matrix = True  # load or build the embedding matrix
	
	if load_embedding_matrix:
		print('loading word vectors...')
		
		# Initialize the embedding matrix with zeros; size is (vocab size + 2, embed_dim)
		# The extra 2 accounts for padding and out-of-vocabulary tokens
		embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
		
		# Set the file path for the word vectors based on the embedding dimension
		# The file contains pre-trained word vectors like GloVe
		fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt'
		
		# Load the word vectors from the file, filtering for the words in word2idx
		word_vec = load_word_vec(fname, word2idx=word2idx)
		print('building embedding_matrix:', embedding_matrix_file_name)
		
		# Loop through the word-to-index mapping
		for word, i in word2idx.items():
			# Get the vector for the word from the loaded word vectors
			vec = word_vec.get(word)
			
			# If the vector exists, place it in the embedding matrix at the appropriate index
			if vec is not None:
				# Words not found in the embedding index will remain as all-zeros in the matrix
				embedding_matrix[i] = vec
		
		# Save the embedding matrix to a file using pickle for later use
		pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
		
	return embedding_matrix

def image_process(image_path, transform):
	
	# Convert to RGB
	image = Image.open(image_path).convert('RGB')
	
	# Apply the specified transformations to the image (e.g., resizing, cropping, normalization)
	image = transform(image)
	return image

class Tokenizer(object):
	def __init__(self, lower=False, max_seq_len=None):
		
		self.lower = lower  # Lowercase text
		self.max_seq_len = max_seq_len  # Maximum length of sequences (text)
		
		# Initialize dictionaries to map words to indices and vice versa
		self.word2idx = {}
		self.idx2word = {}
		
		# Start indexing from 2 (0 and 1 can be reserved for padding and unknown tokens)
		self.idx = 2
		
		# Add a special token 'ttttt' to both mappings with index 1
		self.word2idx['ttttt'] = 1
		self.idx2word[1] = 'ttttt'

	def fit_on_text(self, text):
		if self.lower:
			text = text.lower()  # Convert text to lowercase
		
		words = text.split()  # Split the text into words
		
		# Iterate over the words and add them to the word2idx and idx2word mappings
		for word in words:
			if word not in self.word2idx:
				self.word2idx[word] = self.idx
				self.idx2word[self.idx] = word
				self.idx += 1

	@staticmethod
	def pad_sequence(sequence, maxlen, dtype='int64', padding='pre', truncating='pre', value=0.):
		
		# Create an array of the desired maximum length, filled with the padding value
		x = (np.ones(maxlen) * value).astype(dtype)
		
		# Truncate the sequence if it's longer than maxlen
		if truncating == 'pre':
			trunc = sequence[-maxlen:]  # Truncate from the beginning
		else:
			trunc = sequence[:maxlen]  # Truncate from the end
		
		# Convert the truncated sequence to a numpy array
		trunc = np.asarray(trunc, dtype=dtype)
		
		# Pad the sequence if it's shorter than maxlen
		if padding == 'post':
			x[:len(trunc)] = trunc  # Add padding after the sequence
		else:
			x[-len(trunc):] = trunc  # Add padding before the sequence
		return x

	def text_to_sequence(self, text, reverse=False):
		if self.lower:
			text = text.lower()  # Convert text to lowercase
		words = text.split()  # Split the text into words
		
		# Map each word to its corresponding index, use unknown index for words not in the dictionary
		unknownidx = len(self.word2idx)+1
		sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
		
		# If the sequence is empty, set it to [0] (often used as padding)
		if len(sequence) == 0:
			sequence = [0]
			
		# Set the padding and truncating mode to 'post'
		# Use post padding together with torch.nn.utils.rnn.pack_padded_sequence
		pad_and_trunc = 'post'  # Post padding and truncating
		
		
		# Reverse the sequence if the reverse option is True
		if reverse:
			sequence = sequence[::-1]
			
		# Pad and truncate the sequence to the specified maximum length
		return Tokenizer.pad_sequence(sequence, self.max_seq_len, dtype='int64', padding=pad_and_trunc, truncating=pad_and_trunc)
	

class MVSADataset(Dataset):
	def __init__(self, data):
		# Initialize the dataset with the provided data
		self.data = data

	def __getitem__(self, index):
		# Return a specific data sample based on the index
		return self.data[index]

	def __len__(self):
		# Return the total number of samples in the dataset
		return len(self.data)
	
class MVSADatasetReader:
	@staticmethod
	def __read_text__(fnames):
		# Read text data from the given files
		text = ''
		for fname in fnames:
			with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
				lines = fin.readlines()
				for i in range(1, len(lines)):
					# Extract the text content from each line
					text_raw = lines[i].split('\t')[2].strip().lower()
					text += text_raw + " "
		return text

	@staticmethod
	def __read_data__(fname, tokenizer, path_img, transform):
		print('--------------' + fname + '---------------')
		
		all_data = []
		count = 0
		
		with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
			lines = fin.readlines()

			for i in range(1, len(lines)):
				# Split the line into components based on the dataset structure
				line_parts = lines[i].split('\t')
				
				image_id = line_parts[0].strip()
				sentiment = line_parts[1].strip()
				text_raw = line_parts[2].strip().lower()

				# Tokenize the text
				text_raw_indices = tokenizer.text_to_sequence(text_raw)

				# Convert sentiment to an integer
				sentiment = int(sentiment)

				# Construct the image path
				image_path = os.path.join(path_img, image_id + ".jpg")

				# Load and process the image
				if not os.path.exists(image_path):
					print(f"Image not found: {image_path}")
				try:
					image = image_process(image_path, transform)
				except:
					count += 1
					# Fallback to a default image if the current one fails to load
					image_path_fail = os.path.join(path_img, '0default.jpg')
					image = image_process(image_path_fail, transform)

				# Create a data dictionary for this sample
				data = {
					'text_indices': text_raw_indices,
					'polarity': sentiment,
					'image': image,
				}

				all_data.append(data)

		print('The number of problematic samples: ' + str(count))
		return all_data

	def __init__(self, transform, dataset='mvsa-mts', embed_dim=100, max_seq_len=40, path_image='./images'):
		print("Preparing {0} dataset...".format(dataset))
		
		# Define file paths for the MVSA dataset
		fname = {
			'mvsa-mts': {
				'train': '../../Datasets/MVSA-Modified/mvsa-mts/train.tsv',
				'dev': '../../Datasets/MVSA-Modified/mvsa-mts/val.tsv',
				'test': '../../Datasets/MVSA-Modified/mvsa-mts/test.tsv'
			},
   			'mvsa-mts-100': {
				'train': '../../Datasets/MVSA-Modified/mvsa-mts-100/train.tsv',
				'dev': '../../Datasets/MVSA-Modified/mvsa-mts-100/val.tsv',
				'test': '../../Datasets/MVSA-Modified/mvsa-mts-100/test.tsv'
			}
		}

		# Read the text data from the dataset files to build the tokenizer
		text = MVSADatasetReader.__read_text__([fname[dataset]['train'], fname[dataset]['dev'], fname[dataset]['test']])
		
		# Initialize and fit the tokenizer
		tokenizer = Tokenizer(max_seq_len=max_seq_len)
		tokenizer.fit_on_text(text.lower())
		
		# Build the embedding matrix
		self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
		
		# Prepare the train, dev, and test datasets
		self.train_data = MVSADataset(MVSADatasetReader.__read_data__(fname[dataset]['train'], tokenizer, path_image, transform))
		self.dev_data = MVSADataset(MVSADatasetReader.__read_data__(fname[dataset]['dev'], tokenizer, path_image, transform))
		self.test_data = MVSADataset(MVSADatasetReader.__read_data__(fname[dataset]['test'], tokenizer, path_image, transform))