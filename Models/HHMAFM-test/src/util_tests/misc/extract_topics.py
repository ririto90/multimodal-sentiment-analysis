import os
import re
import nltk
import gensim
import pandas as pd
from gensim import corpora


if not os.path.exists(os.path.join(nltk.data.find('corpora'), 'stopwords')):
    nltk.download('stopwords')
from nltk.corpus import stopwords

# Define stopwords
stop_words = set(stopwords.words('english'))

# Function to preprocess text (remove stopwords, punctuation, etc.)
def preprocess_text(text):
    # Remove URLs, mentions, and special characters
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    
    # Tokenize the text
    tokens = gensim.utils.simple_preprocess(text, deacc=True)  # deacc=True removes punctuations

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Prepare LDA model components (assuming you have preprocessed your corpus already)
def prepare_lda(corpus):
    # Preprocess all texts (convert each tweet into tokens)
    processed_corpus = [preprocess_text(doc) for doc in corpus]

    # Create a dictionary representation of the corpus
    dictionary = corpora.Dictionary(processed_corpus)
    
    # Convert the corpus into a bag-of-words format
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
    
    # Train LDA model
    lda_model = gensim.models.LdaModel(bow_corpus, num_topics=5, id2word=dictionary, passes=10)

    return lda_model, dictionary

# Function to extract topics from an individual tweet
def extract_topics_from_text(tweet, lda_model, dictionary):
    # First, check if there are hashtags in the tweet
    hashtags = re.findall(r'#(\w+)', tweet)

    if hashtags:
        # If hashtags are found, return them as topics (mimicking the left-side process in your figure)
        return hashtags
    else:
        # If no hashtags, preprocess the tweet and apply LDA to extract topics (mimicking the right-side process)
        tokens = preprocess_text(tweet)
        
        # Convert the tokenized tweet into a bag-of-words format
        bow = dictionary.doc2bow(tokens)
        
        # Get the LDA topics with the highest contribution for this tweet
        topics = lda_model.get_document_topics(bow)
        
        # Sort the topics by their weight in descending order
        topics_sorted = sorted(topics, key=lambda x: x[1], reverse=True)
        
        # Extract the top topic's words (you can adjust this to return multiple topics if needed)
        top_words = [lda_model.show_topic(topic[0], topn=5) for topic in topics_sorted[:1]]  # Top 1 topic with 5 words
        
        # Extract just the words (ignoring the probabilities)
        top_words_list = [word for word, prob in top_words[0]]
        
        return top_words_list

# Example usage
if __name__ == "__main__":
    # Example tweets
    files = 'Datasets/MVSA-MTS/mvsa-mts/sentiment.tsv'
    
    mvsa_mts = pd.read_csv('Datasets/MVSA-MTS/mvsa-mts/sentiment.tsv', sep='\t', header=0)
    tweets = mvsa_mts.iloc[10:15]
    print(tweets)
    
    # tweets = [
    #     "Oh yes, we did. #vegan #cheese #torontovegan #food #organic",
    #     "This young woman will not fail to vote in 2015. She knows the importance."
        
    # ]
    print(tweets)

    # Prepare the LDA model using all the tweets (corpus)
    lda_model, dictionary = prepare_lda(tweets)

    # Extract topics for each tweet individually
    for tweet in tweets:
        topics = extract_topics_from_text(tweet, lda_model, dictionary)
        print(f"Tweet: {tweet}\nExtracted Topics: {topics}\n")
