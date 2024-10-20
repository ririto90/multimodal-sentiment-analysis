import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bertopic import BERTopic

# Ensure NLTK data is loaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

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

# Read the dataset
data_path = 'Datasets/MVSA-MTS/mvsa-mts-100/sentiment.tsv'
df = pd.read_csv(data_path, sep='\t', encoding='utf-8')

# Ensure 'Text' column exists
if 'Text' not in df.columns:
    raise ValueError("The 'Text' column was not found in the dataset.")

# Extract the 'Text' column
tweets = df['Text'].astype(str).tolist()

# Preprocess the tweets
processed_tweets = [preprocess_tweet(tweet) for tweet in tweets]

# Initialize BERTopic
topic_model = BERTopic(verbose=True)

# Fit the model
topics, probabilities = topic_model.fit_transform(processed_tweets)

# Get topic information
topic_info = topic_model.get_topic_info()
print("\nTopic Info:")
print(topic_info)

# View topics and their top words
for topic_num in topic_info.Topic.unique():
    if topic_num != -1:  # Ignore outliers
        topic_words = topic_model.get_topic(topic_num)
        print(f"\nTopic {topic_num}:")
        print(', '.join([word for word, _ in topic_words]))

# Assign topics to individual tweets and output them
for idx, (tweet, topic_num) in enumerate(zip(tweets, topics)):
    print(f"\nTweet {idx+1}: {tweet}")
    print(f"Assigned Topic: {topic_num}")
    topic_words = topic_model.get_topic(topic_num)
    print("Topic Keywords:", ', '.join([word for word, _ in topic_words]))
