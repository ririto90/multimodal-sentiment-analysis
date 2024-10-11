import re
import nltk
import string
import pandas as pd
import numpy as np
import umap
import pynndescent

print("NumPy version:", np.__version__)
print("UMAP version:", umap.__version__)
print("PyNNDescent version:", pynndescent.__version__)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bertopic import BERTopic

# Ensure NLTK data is loaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Read the dataset
data_path = 'Datasets/MVSA-MTS/mvsa-mts-100/sentiment.tsv'
df = pd.read_csv(data_path, sep='\t', encoding='utf-8')
tweets = df['Text'].astype(str).tolist()

# Ensure 'Text' column exists
if 'Text' not in df.columns:
    raise ValueError("The 'Text' column was not found in the dataset.")

def preprocess_tweet(tweet):
    tweet = re.sub(r'@\w+', '', tweet)      # Remove mentions
    tweet = re.sub(r'#\w+', '', tweet)      # Remove hashtags
    tweet = re.sub(r'http\S+', '', tweet)   # Remove URLs
    tweet = re.sub(r'[^\w\s]', '', tweet)   # Remove punctuation
    tweet = tweet.lower()
    tokens = word_tokenize(tweet)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def bertopic_on_single_tweet(tweet):
    # Preprocess the tweet
    print(f"Original Tweet: {tweet}")
    processed_tweet = preprocess_tweet(tweet)
    print(f"Processed Tweet: {processed_tweet}")

    # Since BERTopic works better with multiple documents, we'll use the same tweet multiple times
    documents = [processed_tweet]

    # Initialize BERTopic
    topic_model = BERTopic(verbose=False)

    # Fit the model
    topics, probabilities = topic_model.fit_transform(documents)

    # Get the topic information
    topic_info = topic_model.get_topic_info()
    print("\nTopic Info:")
    print(topic_info)

    # Get the topics and their top words
    for topic_num in set(topics):
        if topic_num != -1:  # Ignore outliers labeled as -1
            topic_words = topic_model.get_topic(topic_num)
            print(f"\nTopic {topic_num}:")
            print(', '.join([word for word, _ in topic_words]))
    
    # Since we have only one document, we can get the topic representation for it
    print(f"\nAssigned Topic for the Tweet: {topics[0]}")
    return topics[0], topic_words


# Example tweets
tweet1 = "This young woman will not fail to vote in 2015. She knows the importance."
tweet2 = "This young woman will not fail to vote in 2015. She knows the importance. #woman #vote"
tweet3 = "He will vote in 2015 because he wants his Canada back! #iwillvote2015 #cdnpoli #elxn42 #bcrc15"

print('Tweet 1:')
topic_num, topic_words = bertopic_on_single_tweet(tweet1)

print('\nTweet 2:')
topic_num, topic_words = bertopic_on_single_tweet(tweet2)

print('\nTweet 3:')
topic_num, topic_words = bertopic_on_single_tweet(tweet3)
