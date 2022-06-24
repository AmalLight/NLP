#!/usr/bin/env python

import nltk
import random
import re
import string
import inspect
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus   import twitter_samples
from nltk.corpus   import stopwords
from nltk.stem     import PorterStemmer
from nltk.tokenize import TweetTokenizer

nltk.download ( 'stopwords' )

# import inspect
# print ( inspect.getsource ( build_freqs ) )

print ( 'Define build_freqs , for assign to tweets a category using labels and study its frequency' )

def build_freqs ( tweets , ys ) :

    yslist = np.squeeze(ys).tolist()

    freqs = {}
    
    for y , tweet in zip ( yslist , tweets ) :
    
        for word in process_tweet ( tweet ) :
            pair = ( word , y )
            
            if pair in freqs: freqs [ pair ] += 1
            else:             freqs[pair] = 1

    return freqs

print ( 'Define process_tweet , for pre-processin tweets' )

def process_tweet ( tweet ) :

    stemmer = PorterStemmer ()
    stopwords_english = stopwords.words ( 'english' )
    
    # regex = Regular Expression => to study
    
    # remove stock market tickers like $GE
    tweet = re.sub ( r'\$\w*' , '' , tweet )
    
    # remove old style retweet text "RT"
    tweet = re.sub ( r'^RT[\s]+' , '' , tweet )
    
    # remove hyperlinks
    tweet = re.sub ( r'https?:\/\/.*[\r\n]*' , '' , tweet )
    
    # only removing hashtags # from the word
    tweet = re.sub ( r'#' , '' , tweet )
    
    # tokenize tweets , root ?
    tokenizer = TweetTokenizer ( preserve_case = False , strip_handles = True , reduce_len = True )
    tweet_tokens = tokenizer.tokenize ( tweet )

    tweets_clean = []
    for word in tweet_tokens :
    
        if ( word not in stopwords_english and word not in string.punctuation ) :  # remove punctuation
            stem_word = stemmer.stem ( word )  # stemming word , root ?
            tweets_clean.append ( stem_word )

    return tweets_clean

all_positive_tweets = twitter_samples.strings ( 'positive_tweets.json' )
all_negative_tweets = twitter_samples.strings ( 'negative_tweets.json' )

tweets = all_positive_tweets + all_negative_tweets

print ( 'sample tweet 31:' , tweets [ 31 ] )

m = len ( tweets ) ; print ( "Number of tweets M:" , m )

print ( 'make a numpy array representing labels ( categorical labels ) of the tweets' )

labels = np.append ( np.ones ( ( len ( all_positive_tweets ) ) ) , np.zeros ( ( len ( all_negative_tweets ) ) ) )

print ( "Build complete Dictionary with flags ( of frequencies ) , Labels:" , labels )

freqs = build_freqs ( tweets , labels ) ; example = {}

for (i,key) in zip ( range ( 5 ) , freqs ) :

    example [ key ] = freqs [ key ]

print ( 'Exemple of mini dictionary built before , from 0 to 5:' , example )

print ( 'End of Training' )

# -------------------------------------------------------------------------------------------

print ( 'Starting to Clustering' )

keys = [ 'happi' , 'merri' , 'nice' , 'good' , 'bad' , 'sad' , 'mad' , 'best' , 'pretti' ,
         '❤' , ':)' , ':(' , '😒' , '😬' , '😄' , '😍' , '♛' ,
         'song', 'idea' , 'power', 'play' , 'magnific' ]

# list representing our table of word counts.
# each element consist of a sublist with this pattern: [<word>, <positive_count>, <negative_count>]

data = []

for word in keys:
    
    # initialize positive and negative counts
    pos = 0 ; neg = 0
    
    # retrieve number of positive counts
    if ( word , 1 ) in freqs :
        pos = freqs [ ( word , 1 ) ]
        
    # retrieve number of negative counts
    if ( word , 0 ) in freqs :
        neg = freqs [ ( word , 0 ) ]
        
    # append the word counts to the table
    data.append ( [ word , pos , neg ] )
    
print ( 'data matrix , is like the extract_features ?' , data ) 

# -------------------------------------------------------------------------------------------

print ( 'Visualizing Clustering data' )

fig, ax = plt.subplots ( figsize = ( 13 , 13 ) )

x = np.log ( [ x [ 1 ] + 1 for x in data ] ) # POSITIVE , index 1
y = np.log ( [ x [ 2 ] + 1 for x in data ] ) # NEGATIVE , index 2

ax.scatter ( x , y ) # PLOT

plt.xlabel ( "Log Positive count" )
plt.ylabel ( "Log Negative count" )

# ADD
for i in range ( 0 , len ( data ) ) :
    ax.annotate ( data [ i ] [ 0 ] , ( x [ i ] , y [ i ] ) , fontsize = 12 )

ax.plot ( [ 0 , 9 ] , [ 0 , 9 ] , color = 'red') # Plot , red line => clustering .

plt.show ()

# -------------------------------------------------------------------------------------------

#print ( 'Future' )

#x = np.zeros ( ( m , 3 ) )

#for i in range ( m ) :

    #p_tweet = process_tweet ( tweets [ i ] )
    
    #x [ i , : ] = extract_features ( p_tweet , freqs )

