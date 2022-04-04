
import numpy as np
import re
import pandas as pd
import random
import nltk
from nltk.corpus import stopwords
import requests_oauthlib
from requests_oauthlib import OAuth1Session
from time import sleep
import requests
import json

import tensorflow as tf
tf.__version__  

from tensorflow.keras import layers
import tensorflow_datasets as tfds

# Columns for traning
cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# dataset with 1.6 million tweets  
train_data = pd.read_csv('train.csv', header = None, names = cols, encoding="latin1")

train_data.sentiment.unique()

# Pre-processing 
data = train_data

data.drop(['id', 'date', 'query', 'user'], axis = 1, inplace=True)

data.head()

X = data.iloc[:, 1].values
X

y = data.iloc[:, 0].values
y

# Creating a dataset with 40K tweets
from sklearn.model_selection import train_test_split
X, _, y, _ = train_test_split(X, y, test_size = 0.75, stratify = y)


unique, counts = np.unique(y, return_counts=True)
unique, counts

# Removing stopwords can increase the training and validation accuracy
nltk.download("stopwords")

stop_words = []
for word in stopwords.words('english'):
  stop_words.append(word)

# I did some tests and remove the words bellow from stopwords.
#
# Doing some tests here can increase accuracy.
stop_words_out = ["no","nor","not","don","don't","ain","aren","aren't",\
                  "couldn","couldn't","didn","didn't","doesn","doesn't",\
                  "isn","isn't","wasn","wasn't","weren","weren't"]

stop_words = [word for word in stop_words  if word not in stop_words_out]


# Cleaning the tweets
# Do some tests here, it can also encrease the accuracy.
def clean_tweets(tweet):
  tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
  tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
  tweet = re.sub(r"[0-9]", '', tweet)
  tweet = re.sub(r"[^a-zA-Z!?,'.:;`]", ' ', tweet)
  tweet = re.sub(r"[_-]+", '', tweet)
  tweet = re.sub(r'[\?\.\!|.\ ]+(?=[\?\.\!|. ]{2})', '', tweet)
  tweet = re.sub(r'(\w)\1+', r'\1', tweet)
  tweet =  re.sub(r"\b [a-zA-Z]\b", "", tweet)
   
  tweet = [word.lower() for word in tweet.split() if word not in stop_words]
  
  tweet = ' '.join([str(element) for element in tweet])
  
  return tweet

# Testing the function
text = "i like you"
clean_tweets(text)

text = "i hate you"
clean_tweets(text)

# Cleaning all tweets
data_clean = [clean_tweets(tweet) for tweet in X]

# Getting the label
data_labels = y

# Changing labels to 1 and 0 (Positive, Negative)
data_labels[data_labels == 4] = 1

# Token
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(data_clean, target_vocab_size=2**16)

data_inputs = [tokenizer.encode(sentence) for sentence in data_clean]

# Padding
max_len = max([len(sentence) for sentence in data_inputs])

data_inputs = tf.keras.preprocessing.sequence.pad_sequences(data_inputs,
                                                            value = 0,
                                                            padding = 'post',
                                                            maxlen=max_len)



# Spliting the dataset into training and test
train_inputs, test_inputs, train_labels, test_labels = train_test_split(data_inputs,
                                                                        data_labels,
                                                                        test_size=0.3,
                                                                        stratify = data_labels)

# Building the model
class DCNN(tf.keras.Model):

  def __init__(self,
               vocab_size,
               emb_dim=128,
               nb_filters=50,
               ffn_units=512,
               nb_classes=2,
               dropout_rate=0.1,
               training=True,
               name="dcnn"):
    super(DCNN, self).__init__(name=name)

    self.embedding = layers.Embedding(vocab_size, emb_dim)

    self.bigram = layers.Conv1D(filters=nb_filters, kernel_size=2, padding='same', activation='relu')

    self.trigram = layers.Conv1D(filters=nb_filters, kernel_size=3, padding='same', activation='relu')

    self.fourgram = layers.Conv1D(filters=nb_filters, kernel_size=4, padding='same', activation='relu')

    self.pool = layers.GlobalMaxPool1D()

    self.dense_1 = layers.Dense(units = ffn_units, activation = 'relu')
    
    self.dropout = layers.Dropout(rate = dropout_rate)
    
    self.last_dense = layers.Dense(units = 1, activation = 'sigmoid')
    

  def call(self, inputs, training):
    x = self.embedding(inputs)
    x_1 = self.bigram(x)
    x_1 = self.pool(x_1)
    x_2 = self.trigram(x)
    x_2 = self.pool(x_2)
    x_3 = self.fourgram(x)
    x_3 = self.pool(x_3)

    merged = tf.concat([x_1, x_2, x_3], axis = -1) # (batch_size, 3 * nb_filters)
    merged = self.dense_1(merged)
    merged = self.dropout(merged, training)
    output = self.last_dense(merged)

    return output

    # Parameters configuration
    vocab_size = tokenizer.vocab_size
    vocab_size

    emb_dim = 200
    nb_filters = 100
    ffn_units = 256
    batch_size = 64
    nb_classes = len(set(train_labels))
    nb_classes

    dropout_rate = 0.2

    # Short training 
    nb_epochs = 5

    # Training 
    Dcnn = DCNN(vocab_size=vocab_size, emb_dim=emb_dim, nb_filters=nb_filters,
              ffn_units=ffn_units, nb_classes=nb_classes, dropout_rate=dropout_rate)

    Dcnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint_path = "./"
    ckpt = tf.train.Checkpoint(Dcnn=Dcnn)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored')

    history = Dcnn.fit(train_inputs, train_labels,
                     batch_size = batch_size,
                     epochs = nb_epochs,
                     verbose = 1,
                     validation_split = 0.10)
    ckpt_manager.save()

    # Training dataset: 76% accuracy
    #
    # Model evaluation
    results = Dcnn.evaluate(test_inputs, test_labels, batch_size=batch_size)
    print(results)
    # Test dataset: 76% accuracy

    y_pred_test = Dcnn.predict(test_inputs)

    y_pred_test = (y_pred_test > 0.5)


    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(test_labels, y_pred_test)

    # Result confusion matrix
    cm

    # Twitter Authentication
    # You can obtain your keys on 'developer.twitter.com/'
    key = 'Your consumer key'
    key_secret = 'Your consumer secret'

    token = 'Your access token'
    token_secret = 'Your access token secret'

    search = 'vaccine'

    url = 'https://stream.twitter.com/1.1/statuses/sample.json'
    url_search = 'https://stream.twitter.com/1.1/statuses/filter.json?track='+search+'&lang=en'

    auth = requests_oauthlib.OAuth1(key, key_secret, token, token_secret)

    # Number of tweets in a batch
    tweets_num = 100


def stream_twitter():
  """Makes the conexion with Tweeter API"""
  response = requests.get(url_search, auth = auth, stream = True)
  if response.status_code == 200:
      print("\nGetting tweets...wait...")
  else:
      print("you are not conected")
  for line in response.iter_lines():
    try:
      post = json.loads(line.decode('utf-8'))
      contents = [post['text']]
      yield str(contents)
    except:
      return False
 

def twitter_list(stream_twitter):
    """
    This function creates the Batch for analysis

    Keywords argument:
    stream_twitter -- Function that create the conexions.
    """
    count=0
    list_stream =[]
    for i in stream_twitter:
        if count >= tweets_num:
            break
        if search in i and i not in list_stream:
            list_stream.append(i)
            count+=1
    return list_stream


# Tweet Analysis
def twitter_sentiment(twitter_list):
    n=1
    positive = 0
    negative = 0
    
    for i in twitter_list:
        print("tweet {}\n{}".format(n,i))
        n+=1
        text = tokenizer.encode(clean_tweets(i))
        result = Dcnn(np.array([text]), training=False).numpy()
        if result > 0.5:
            positive += 1
            print('\nPositive tweet\n')
        else:
            negative += 1
            print('\nNegative tweet\n')
    print( 'Last {} tweets:\n' 'Positive: {}\nNegative: {}\n'.format(tweets_num, positive, negative))
    return positive, negative
    sleep(1)


def main():
    positive = 0
    negative = 0
    while True: 
        try:
            list_twitter = twitter_list(stream_twitter())
            twitter_sentiment(list_twitter)
            
            positive_twitter, negative_twitter = twitter_sentiment(list_twitter)
            
            positive += positive_twitter
            negative += negative_twitter
            
            print("Sentiment about {}, total: Positive: {} --- Negative: {}".format(search, positive, negative))
            sleep(1)
        except Exception as error:
                print("Error, {}".format(error))
                break
       

if __name__ == '__main__':
    main()
