import sys
import os
import collections

# Add the parent directory to sys.path
sys.path.append("/Users/roneng100/Downloads/Multimodal-Sentiment-Analysis-main/Models/ESAFN-modified")

# Use absolute imports
from data_utils import MVSADatasetReader

import torchvision.transforms as transforms

def count_and_print_sentiments(data, dataset_name):
    # Initialize a counter for the sentiment polarities
    sentiment_counter = collections.Counter()

    # Count sentiment polarities
    for sample in data:
        sentiment_counter[sample['polarity']] += 1

    # Print the sentiment counts
    print(f"\nSentiment distribution in {dataset_name} set:")
    for sentiment, count in sentiment_counter.items():
        print(f"Sentiment {sentiment}: {count} samples")

    # Print details for the first few samples
    for i in range(3):
        sample = data[i]
        print(f"\nSample {i + 1}:")
        print("Text indices:", sample['text_raw_indices'])
        print("Sentiment:", sample['polarity'])
        print("Image tensor shape:", sample['image'].shape)

def test_mvsa_dataset_reader():
    # Define the transformations for the images (can be simple for testing)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images for simplicity
        transforms.ToTensor(),          # Convert images to PyTorch tensors
    ])

    # Initialize the dataset reader for the 'mvsa' dataset
    dataset_reader = MVSADatasetReader(
        transform=transform,
        dataset='mvsa-mts-100',
        embed_dim=100,
        max_seq_len=40,
        path_image='../../Datasets/MVSA-Modified/images'  # Ensure this path contains some test images
    )

    # Print some details about the embedding matrix
    print("Embedding matrix shape:", dataset_reader.embedding_matrix.shape)

    # Process the training dataset
    train_data = dataset_reader.train_data
    print("\nNumber of training samples:", len(train_data))
    count_and_print_sentiments(train_data, "training")

    # Process the validation dataset
    val_data = dataset_reader.dev_data
    print("\nNumber of validation samples:", len(val_data))
    count_and_print_sentiments(val_data, "validation")

    # Process the test dataset
    test_data = dataset_reader.test_data
    print("\nNumber of test samples:", len(test_data))
    count_and_print_sentiments(test_data, "test")

# Run the test
if __name__ == "__main__":
    test_mvsa_dataset_reader()
