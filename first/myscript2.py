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

import pandas as pd # Library for Dataframes 


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

print ( 'split the data into two pieces, one for training and one for testing ( validation set )' )

train_pos = all_positive_tweets [ : 4000 ]
train_neg = all_negative_tweets [ : 4000 ]

test_pos = all_positive_tweets [ 4000 ]
test_neg = all_negative_tweets [ 4000 ]

train = train_pos + train_neg
test  =  test_pos +  test_neg

print ( "Number of tweets:" , len ( train ) )

# -------------------------------------------------------------------------------------------

print ( 'Load logistic_features csv using pandas function' )

data = pd.read_csv ( 'logistic_features.csv' )

print ( 'Print the first 10 data entries' )

data.head ( 10 )

print ( data )

print ( 'Each feature is labeled as bias=theta ? , positive and negative' )

X = data [ [ 'bias' , 'positive' , 'negative' ] ] . values

Y = data [ 'sentiment' ] . values

print ( 'Print the shape of the X part' )
print ( X.shape )

print ( '# Print some rows of X' )
print ( X )

# -------------------------------------------------------------------------------------------

print ( 'Load a pretrained Logistic Regression model' )

theta = [ 7e-08 , 0.0005239 , -0.00055517 ]

print ( 'Plot the samples using columns 1 and 2 of the matrix' )

fig, ax = plt.subplots ( figsize = ( 8 , 8 ) )

colors = [ 'red' , 'green' ]

print ( 'Color based on the sentiment Y' )

ax.scatter ( X [ : , 1 ] , X [ : , 2 ] , c = [ colors [ int ( k ) ] for k in Y ] , s = 0.1 ) # Plot a dot for each pair of words

plt.xlabel ( "Positive" )
plt.ylabel ( "Negative" )

# -------------------------------------------------------------------------------------------

# Equation for the separation plane
# It give a value in the negative axe as a function of a positive value
# f(pos, neg, W) = w0 + w1 * pos + w2 * neg = 0

print (
    's(pos, W) = (w0 - w1 * pos) / w2' ,
    'this because theta is a big logit',
    'with value between 0 and 1'
)

def neg ( theta , pos ) : return ( -theta[0] - pos * theta[1] ) / theta[2]

# Equation for the direction of the sentiments change
# We don't care about the magnitude of the change. We are only interested 
# in the direction. So this direction is just a perpendicular function to the 
# separation plane

print (
    'df(pos, W) = pos * w2 / w1',
    'this because theta is a big logit',
    'with value between 0 and 1'
)

def direction ( theta , pos ) : return pos * theta[2] / theta[1]

# -------------------------------------------------------------------------------------------

# Plot the samples using columns 1 and 2 of the matrix

fig, ax = plt.subplots ( figsize = ( 8 , 8 ) )

colors = ['red', 'green']

# Color base on the sentiment Y

ax.scatter ( X [ : , 1 ] , X [ : , 2 ] , c = [ colors [ int ( k ) ] for k in Y ] , s = 0.1 ) # Plot a dot for each pair of words

plt.xlabel ( "Positive" )
plt.ylabel ( "Negative" )

# Now lets represent the logistic regression model in this chart.

maxpos = np.max ( X [ : , 1 ] )

offset = 5000 # The pos value for the direction vectors origin

# Plot a gray line that divides the 2 areas.

ax.plot ( [ 0 ,  maxpos ] , [ neg ( theta , 0 ) , neg ( theta , maxpos ) ] , color = 'gray')

# Plot a green line pointing to the positive direction

ax.arrow ( offset , neg ( theta , offset ) , offset , direction ( theta , offset ) , head_width = 500 , head_length = 500 , fc = 'g', ec = 'g' )

# Plot a red line pointing to the negative direction

ax.arrow ( offset , neg ( theta , offset ) , -offset , -direction ( theta , offset ) , head_width = 500 , head_length = 500 , fc = 'r' , ec = 'r' )

plt.show ()
