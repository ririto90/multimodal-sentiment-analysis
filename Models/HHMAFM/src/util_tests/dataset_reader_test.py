# test_mvsa_dataset_reader.py
import sys
import os
from util_tests.data_utils_test import MVSADatasetReader
from torchvision import transforms

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

def main():
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Set the path to the images
    path_image = 'Datasets/MVSA-MTS/images'  # Update this path to your images directory
    
    # Initialize the dataset reader
    dataset_reader = MVSADatasetReader(
        transform=transform,
        dataset='mvsa-mts-100',    # Choose between 'mvsa-mts' or 'mvsa-mts-100'
        embed_dim=100,
        max_seq_len=40,
        path_image=path_image
    )
    
    # Access the train, dev, and test datasets
    train_data = dataset_reader.train_data
    dev_data = dataset_reader.dev_data
    test_data = dataset_reader.test_data
    
    # Print out the lengths of the datasets
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of development samples: {len(dev_data)}")
    print(f"Number of test samples: {len(test_data)}")
    
    # Print out a sample from the train data
    sample = train_data[0]
    print("\nSample data from training set:")
    print(f"Text: {sample['raw_text']}")
    print(f"Topic: {sample['raw_topic']}")
    # print(f"Text indices: {sample['text_indices']}")
    # print(f"Topic indices: {sample['topic_indices']}")
    print(f"input_ids_text: {sample['input_ids_text'].squeeze(0)}")
    print(f"attention_mask_text: {sample['attention_mask_text'].squeeze(0)}")
    print(f"input_ids_topic: {sample['input_ids_topic'].squeeze(0)}")
    print(f"attention_mask_topic: {sample['attention_mask_topic'].squeeze(0)}")
    print(f"Polarity: {sample['polarity']}")
    print(f"Image shape: {sample['image'].shape}")
    

if __name__ == '__main__':
    main()
