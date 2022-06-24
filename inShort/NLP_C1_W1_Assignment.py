#!/usr/bin/env python

import nltk
from os import getcwd

nltk.download('twitter_samples')
nltk.download('stopwords')

from utils import process_tweet
from utils import build_freqs

# -----------------------------------------------------

filePath = f"{getcwd()}/../tmp2/"
nltk.data.path.append(filePath)

import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples 

from utils import process_tweet, build_freqs

# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# -----------------------------------------------------

# split the data into two pieces, one for training and one for testing (validation set) 
print ( 'range for all_positive_tweets:' , len ( all_positive_tweets ) , ',' , '80%:' , len ( all_positive_tweets ) * 0.80 )
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]

print ( 'range for all_negative_tweets:' , len ( all_negative_tweets ) , ',' , '80%:' , len ( all_negative_tweets ) * 0.80 )
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg 
test_x  =  test_pos +  test_neg

# combine positive and negative labels
train_y = np.append ( np.ones ( ( len ( train_pos ) , 1 ) ) , np.zeros ( ( len ( train_neg ) , 1 ) ) , axis = 0 )
test_y  = np.append ( np.ones ( ( len (  test_pos ) , 1 ) ) , np.zeros ( ( len (  test_neg ) , 1 ) ) , axis = 0 )

# Print the shape train and test sets
print ( "train_y.shape = " + str ( train_y.shape ) )
print ( "test_y.shape  = " + str ( test_y.shape  ) )

# -----------------------------------------------------

# create frequency dictionary
freqs = build_freqs(train_x, train_y)

# check the output
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))

# test the function below
print('This is an example of a positive tweet: \n', train_x[0])
print('\nThis is an example of the processed version of the tweet: \n', process_tweet(train_x[0]))

# -----------------------------------------------------

# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def sigmoid ( z ) :
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # calculate the sigmoid of z
    h = 1 / ( 1+ np.exp(-z) )
    ### END CODE HERE ###
    
    return h

# Testing your function 
if (sigmoid(0) == 0.5):
    print('SUCCESS!')
else:
    print('Oops!')

if (sigmoid(4.92) == 0.9927537604041685):
    print('CORRECT!')
else:
    print('Oops again!')

# -----------------------------------------------------

# verify that when the model predicts close to 1, but the actual label is 0, the loss is a large positive value
-1 * (1 - 0) * np.log(1 - 0.9999) # loss is about 9.2

# y = 0 && predict close to 1
# -1 * ( ( 0 * ... ) + 1-0 * log( 1 - 0.9999) )
# -1 * ( 1 * log ( 0.0001 ) ) = -1 * -9.2 = 9.2

# predict close to 0 && y = 1
# -1 * ( ( 1 * log ( 0.0001 ) ) + ( 1-1 * ... ) )
# -1 * ( -9.2 ) = 9.2

# verify that when the model predicts close to 0 but the actual label is 1, the loss is a large positive value
-1 * np.log(0.0001) # loss is about 9.2

# -----------------------------------------------------

# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def gradientDescent(x, y, theta, alpha, num_iters):

    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    Hint: you might want to print the cost to make sure that it is going down.
    '''
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # get 'm', the number of rows in matrix x
    
    m = len ( x )
    a = alpha
    J = 0
            
    for i in range ( 0 , num_iters ):
        
        # get z, the dot product of x and theta
        z = np.dot( x , theta )
                
        # get the sigmoid of z
        h = sigmoid ( z )
        
        # calculate the cost function
        logh     = np.𝑙𝑜𝑔 (      𝐡 )
        logh_min = np.𝑙𝑜𝑔 ( 1. - 𝐡 )
        
        newy = 1.-y
        
        left  =    y * 𝑙𝑜𝑔h
        right = newy * 𝑙𝑜𝑔h_min
        
        J = float ( sum ( left + right ) / m * - 1. )        
        
        # np.multiply ( x , y ) == x * y

        # update the weights theta
        theta = theta - ( ( np.dot ( np.transpose ( x ) , h - y ) ) * alpha / m )
        
    ### END CODE HERE ###
    
    J = float ( J )
    return J , theta
    
np.random.seed ( 1 )
tmp_X = np.append ( np.ones ( ( 10 , 1 ) ) , np.random.rand( 10 , 2 ) * 2000 , axis=1 )
tmp_Y = ( np.random.rand ( 10 , 1 ) > 0.35 ).astype ( float )

# Apply gradient descent

tmp_J, tmp_theta = gradientDescent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)

print(f"The cost after training is {tmp_J:.8f}.")

print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}")

# -----------------------------------------------------

# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def extract_features(tweet, freqs):
    '''
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(tweet)
    
    # 3 elements in the form of a 1 x 3 vector = Theta
    x = np.zeros((1, 3)) 
    
    #bias term is set to 1
    x[0,0] = 1 
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # loop through each word in the list of words
    for word in word_l:
        
        if ( ( word , 1 ) in freqs.keys() ): x[0,1] += float ( freqs.get ( ( word , 1 ) ) )
        if ( ( word , 0 ) in freqs.keys() ): x[0,2] += float ( freqs.get ( ( word , 0 ) ) )
        
    ### END CODE HERE ###
    
    assert ( x.shape == ( 1 , 3 ) )
    return x

# test 1
# test on training data
tmp1 = extract_features(train_x[0], freqs)
print(tmp1)

# test 2:
# check for when the words are not in the freqs dictionary
tmp2 = extract_features('blorb bleeeeb bloooob', freqs)
print(tmp2)

-----------------------------------------------------


# ## Part 3: Training Your Model

# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Apply gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

# -----------------------------------------------------

# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def predict_tweet(tweet, freqs, theta):
    '''
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output: 
        y_pred: the probability of a tweet being positive or negative
    '''
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # extract the features of the tweet and store it into x
    x = extract_features ( tweet , freqs )
    
    # make the prediction using x and theta
    y_pred = sigmoid ( np.dot ( x , theta ) )
    
    ### END CODE HERE ###
    
    return y_pred

# Run this cell to test your function
for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    print( '%s -> %f' % (tweet, predict_tweet(tweet, freqs, theta)))


# **Expected Output**: 
# ```
# I am happy -> 0.518580
# I am bad -> 0.494339
# this movie should have been great. -> 0.515331
# great -> 0.515464
# great great -> 0.530898
# great great great -> 0.546273
# great great great great -> 0.561561
# ```

# Feel free to check the sentiment of your own tweet below
my_tweet = 'I am learning :)'
predict_tweet(my_tweet, freqs, theta)

# -----------------------------------------------------

# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Input: 
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """
    
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # the list for storing predictions
    m = len ( test_y )
    
    y_hat = np.zeros ( (m,1) )
    
    #print ( test_x      )
    #print ( test_x[ 0 ] )
    
    for i in range ( m ) :
        
        # get the label prediction for the tweet
        y_pred = predict_tweet ( test_x [ i ] , freqs , theta )
        
        if y_pred > 0.5: y_hat [ i,0 ] = 1
        else:            y_hat [ i,0 ] = 0

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    accuracy = 0
    
    for i in range ( m ):
        if y_hat [ i,0 ] == test_y [ i,0 ]:
            accuracy += 1
            
    accuracy = accuracy / m

    ### END CODE HERE ###
    
    return accuracy

tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")

# -----------------------------------------------------

# Feel free to change the tweet below
my_tweet = 'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!'

print(process_tweet(my_tweet))
y_hat = predict_tweet(my_tweet, freqs, theta)

print(y_hat)

if y_hat > 0.5:
    print('Positive sentiment')
else: 
    print('Negative sentiment')

