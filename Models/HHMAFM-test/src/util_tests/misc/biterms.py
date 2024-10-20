import re
import nltk
import string
from nltk.tokenize import word_tokenize
from biterm.utility import vec_to_biterms, topic_summuary
from biterm.btm import oBTM
from sklearn.feature_extraction.text import CountVectorizer

# Ensure nltk data is loaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(nltk.corpus.stopwords.words('english'))

# Example tweets
tweets = [
    "This young woman will not fail to vote in 2015. She knows the importance.",
    "This young woman will not fail to vote in 2015. She knows the importance. #woman #vote",
    "He will vote in 2015 because he wants his Canada back! #iwillvote2015 #cdnpoli #elxn42 #bcrc15"
]

# Preprocess tweets
def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'#\w+', '', tweet)  # Remove hashtags
    tweet = re.sub(r'http\S+', '', tweet)  # Remove URLs
    tweet = re.sub(r'@\w+', '', tweet)  # Remove mentions
    tweet = re.sub(r'[^\w\s]', '', tweet)  # Remove punctuation
    tokens = word_tokenize(tweet)
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)

processed_tweets = [preprocess_tweet(tweet) for tweet in tweets]

# Vectorize the documents
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_tweets).toarray()
vocab = np.array(vectorizer.get_feature_names_out())

# Create biterms
from biterm.utility import vec_to_biterms
biterms = vec_to_biterms(X)

# Instantiate and fit the BTM model
btm = oBTM(num_topics=2, V=vocab)
btm.fit(biterms, iterations=50)

# Transform the documents to topic distributions
doc_topics = btm.transform(X)

# Display the topics
for i in range(btm.K):
    print(f"\nTopic {i}:")
    topic_words = btm.topic_words(i, top_n=10)
    print(', '.join(topic_words))
