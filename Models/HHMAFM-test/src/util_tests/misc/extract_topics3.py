import re
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Ensure nltk data is loaded
nltk.data.path.append('/usr/local/Caskroom/miniconda/base/envs/ml/nltk_data')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

# Function to preprocess the tweet
def preprocess_tweet(tweet):
    # stop_words = set(stopwords.words('english'))
    tweet = re.sub(r'#\w+', '', tweet) # removes hastags uncomment/comment
    tokens = word_tokenize(tweet.lower())
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [re.sub(r'[^\w\s]', '', word) for word in tokens if word.strip()]
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words and word not in string.punctuation]
    return tokens

# Function to perform LDA on a single tweet and output word importance
def lda_on_single_tweet(tweet):
    
    # hashtags = re.findall(r'#(\w+)', tweet)
    # if hashtags:
    #     # If hashtags are found, return them as topics
    #     print(hashtags)
    #     return hashtags
    # else:
    # Preprocess the tweet
    print(tweet)
    processed_tweet = preprocess_tweet(tweet)
    print(processed_tweet)

    # Create dictionary and corpus for the tweet
    dictionary = corpora.Dictionary([processed_tweet])
    corpus = [dictionary.doc2bow(processed_tweet)]

    # Perform LDA on the tweet with 1 topic (since it's just 1 tweet)
    lda_model = LdaModel(corpus, num_topics=1, id2word=dictionary, random_state=42)
    print(lda_model)

    # Output the words and their importance for the single topic
    print(f"Words and their weights for the tweet:")
    for idx, topic in lda_model.print_topics(num_topics=5, num_words=5):  # Show top 5 words
        print(f"Topic {idx + 1}: {topic}")

    # Print the simplified output format (similar to the image)
    important_words = [word for word, prob in lda_model.show_topic(0, topn=len(dictionary))] # len(dictionary) shows all 
    print(f"Simplified Output: {', '.join(important_words)}")
    return important_words

# Example tweet (similar to the one in your image)
tweet1 = "This young woman will not fail to vote in 2015. She knows the importance."
tweet2 = "This young woman will not fail to vote in 2015. She knows the importance. #woman #vote"
tweet3 = "he will vote in 2015 because he wants his canada back! #iwillvote2015 #cdnpoli #elxn42 #bcrc15"

# Run the LDA function on the example tweet
print('Tweet 1:')
topics = lda_on_single_tweet(tweet1)
print(topics)

print('Tweet 2:')
topics = lda_on_single_tweet(tweet2)
print(topics)

print('Tweet 3:')
topics = lda_on_single_tweet(tweet3)
print(topics)
