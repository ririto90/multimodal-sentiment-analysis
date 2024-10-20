import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
nltk.download('punkt')

# Load stop words
stop_words = set(stopwords.words('english'))

# Load the dataset
df = pd.read_csv('Datasets/MVSA-MTS/mvsa-mts-100/sentiment.tsv', sep='\t', header=0)

# Extract tweets
tweets = df['Text'][:10]  # Replace 'tweet_column_name' with the actual name of the column that contains the tweets

# Preprocess the tweets
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetical characters
    tokens = re.findall(r'\b\w+\b', text)  # Tokenize using regex to avoid NLTK Punkt issues
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

preprocessed_tweets = [preprocess_text(tweet) for tweet in tweets]

# Create an LDA model
lda_models = []
tweet_topics = []

for tweet_tokens in preprocessed_tweets:
    if len(tweet_tokens) > 0:
        dictionary = corpora.Dictionary([tweet_tokens])
        corpus = [dictionary.doc2bow(tweet_tokens)]
        lda_model = LdaModel(corpus, num_topics=min(5, len(dictionary)), id2word=dictionary, passes=15)
        lda_models.append(lda_model)
        topics = lda_model.show_topics(num_topics=min(5, len(dictionary)), num_words=5, formatted=False)
        tweet_topics.append(topics)
    else:
        lda_models.append(None)
        tweet_topics.append([])

# Output the most relevant topic for each tweet
for i, tweet in enumerate(tweets):
    print(f'Tweet: {tweet}')
    topics = tweet_topics[i]
    if topics:
        for topic_num, prob in topics:
            if isinstance(prob, (int, float)):
                topic_words = lda_models[i].show_topic(topic_num, topn=1)
                print(topic_words)
                word = topic_words[0][0]
                print(f'Topic {topic_num + 1}: {word} ({prob:.2f})')
    print('-' * 40)