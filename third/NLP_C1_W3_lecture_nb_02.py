#!/usr/bin/env python

import pandas as pd
import numpy  as np
import pickle

import json
from   json import JSONEncoder
import numpy

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def savePictures( fig , name ) : return fig.savefig ( '{}.png'.format( name ) )

def difference ( list1 , list2 ) :

    len1 = len ( list1 )
    
    list3 = []
    
    if not len1 == len ( list2 ) : return list1
    
    else:
    
        for i in range ( len1 ) : list3 += [ list1 [ i ] - list2 [ i ] ]
        
        return list3

# --------------------------------------------------------------------------------

# Opening JSON file

f = open ( 'word_embeddings_subset.json' , 'r' )
  
# returns JSON object as a dictionary

word_embeddings = json.load ( f )

#print ( word_embeddings )

f.close ()

# --------------------------------------------------------------------------------

#word_embeddings = pickle.load( open( "word_embeddings_subset.p", "rb" ) )

print ( 'there should be 243 words that will be used in this assignment' )

len(word_embeddings)

# --------------------------------------------------------------------------------

print ( 'Get the vector representation for the word country' )

countryVector = word_embeddings['country']

print ( 'Print the type of the vector. Note it is a numpy array' )

print(type(countryVector))

print ( 'Print the short values of the vector' )

print ( countryVector [ 0:5 ] )

print ( 'It is important to note that we store each vector as a NumPy array. It allows us to use the linear algebra operations on it.' )
 
print ( 'The vectors have a size of 300, while the vocabulary size of Google News is around 3 million words!' )

print ( 'Define a get function for get the vector for a given word' )

def vec ( w ) : return word_embeddings [ w ]

print ( '' )
# --------------------------------------------------------------------------------

print ( 'Operating on word embeddings' )

print ( 'These word embedding needs to be validated or at least understood because the performance of the derived model will strongly depend on its quality.' )
 
print ( 'Word embeddings are multidimensional arrays, usually with hundreds of attributes that pose a challenge for its interpretation.' )
 
print ( 'In this notebook, we will visually inspect the word embedding of some words using a pair of attributes.' )

print ( 'Raw attributes are not the best option for the creation of such charts but will allow us to illustrate the mechanical part in Python. ' )
 
print ( 'In the next cell, we make a beautiful plot for the word embeddings of some words.' )

print ( 'Even if plotting the dots gives an idea of the words, the arrow representations help to visualize the vector\'s alignment as well.' )

words = [ 'oil' , 'gas' , 'happy' , 'sad' , 'city' , 'town' , 'village' , 'country' , 'continent' , 'petroleum' , 'joyful' ]

print ( 'Convert each word to its vector representation' )

bag2d = np.array ( [ vec ( word ) for word in words ] )

# ---------------------------------------------------------------------------------------------

print ( 'Create custom size image' )

fig, ax = plt.subplots ( figsize = ( 10 , 10 ) )

print ( 'Select the column for the x axis' )

col1 = 3

print ( 'Select the column for the y axis' )

col2 = 2

print ( 'Print an arrow for each word' )

for word in bag2d : ax.arrow ( 0 , 0 , word [ col1 ] , word [ col2 ] , head_width=0.005 , head_length=0.005 , fc='r' , ec='r' , width=1e-5 )

print ( 'Plot a dot for each word' )
    
ax.scatter ( bag2d [ : , col1 ] , bag2d [ : , col2 ] )

print ( 'Add the word label over each dot in the scatter plot' )

for i in range ( 0 , len ( words ) ) : ax.annotate ( words [ i ] , ( bag2d [ i , col1 ] , bag2d [ i , col2 ] ) )

# plt.show()
savePictures ( fig , 'word_embeddings_rappresentation_1' )

print ( 'Note that similar words like \'village\' and \'town\' or \'petroleum\', \'oil\', and \'gas\' tend to point in the same direction.' )

print ( 'Also, note that \'sad\' and \'happy\' looks close to each other; however, the vectors point in opposite directions.' )

print ( 'In this chart, one can figure out the angles and distances between the words. Some words are close in both kinds of distance metrics.' )

print ( '' )
# ---------------------------------------------------------------------------------------------

print ( 'Now plot the words \'sad\', \'happy\', \'town\', and \'village\'.' )

print ( 'In this same chart, display the vector from \'village\' to \'town\' and the vector from \'sad\' to \'happy\'. Let us use NumPy for these linear algebra operations.' )

words = ['sad', 'happy', 'town', 'village']

print ( 'Convert each word to its vector representation' )

bag2d = np.array ( [ vec ( word ) for word in words ] )

print ( 'Create custom size image' )

fig, ax = plt.subplots ( figsize = ( 10 , 10 ) )

print ( 'Select the column for the x axe' )

col1 = 3

print ( 'Select the column for the y axe' )

col2 = 2

print ( 'Print an arrow for each word' )

for word in bag2d : ax.arrow (0 , 0 , word [ col1 ] , word [ col2 ] , head_width=0.0005 , head_length=0.0005 , fc='r' , ec='r' , width=1e-5 )
    
print ( 'print the vector difference between village and town' )

village = vec ( 'village' )
town    = vec ( 'town'    )

diff = difference ( town , village )

ax.arrow ( village [ col1 ] , village [ col2 ] , diff [ col1 ] , diff [ col2 ] , fc='b', ec='b', width=1e-5 )

print ( 'print the vector difference between village and town' )

sad   = vec ( 'sad'   )
happy = vec ( 'happy' )

diff = difference ( happy , sad )

ax.arrow ( sad [ col1 ] , sad [ col2 ] , diff [ col1 ] , diff [ col2 ] , fc='b' , ec='b' , width=1e-5 )

print ( 'Plot a dot for each word' )

ax.scatter ( bag2d [ : , col1 ] , bag2d [ : , col2 ] )

print ( 'Add the word label over each dot in the scatter plot' )

for i in range ( 0 , len ( words ) ): ax.annotate ( words [ i ] , ( bag2d [ i , col1 ] , bag2d [ i , col2 ] ) )

# plt.show()
savePictures ( fig , 'word_embeddings_rappresentation_2' )

print ( '' )
# ---------------------------------------------------------------------------------------------

print ( 'Linear algebra on word embeddings' )
 
print ( 'In the lectures, we saw the analogies between words using algebra on word embeddings. Let us see how to do it in Python with Numpy.' )
 
print ( 'To start, get the **norm** of a word in the word embedding.' )

print ( 'Print the norm of the word town' )

print ( np.linalg.norm ( vec ( 'town' ) ) )

print ( 'Print the norm of the word sad' )

print ( np.linalg.norm ( vec ( 'sad' ) ) )

print ( '' )
# ---------------------------------------------------------------------------------------------

print ( 'Predicting capitals' )
 
print ( 'Now, applying vector difference and addition, one can create a vector representation for a new word.' )

print ( 'For example, we can say that the vector difference between \'France\' and \'Paris\' represents the concept of Capital.' )

print ( 'One can move from the city of Madrid in the direction of the concept of Capital.' )

print ( 'and obtain something close to the corresponding country to which Madrid is the Capital.' )

capital = difference ( vec ( 'France' ) , vec ( 'Paris' ) )
country = difference ( vec ( 'Madrid' ) , capital )

print ( 'Print the first 5 values of the vector' )

print ( country [ 0 : 5 ] )

print ( 'We can observe that the vector \'country\' that we expected to be the same as the vector for Spain is not exactly it.' )

diff = difference ( country , vec ( 'Spain' ) )

print ( diff [ 0 : 10 ] )

print ( 'If the word embedding works as expected, the most similar word must be \'Spain\'.' )

print ( 'Let us define a function that helps us to do it. We will store our word embedding as a DataFrame, which facilitate the lookup operations based on the numerical vectors.' )

print ( 'Create a dataframe out of the dictionary embedding. This facilitate the algebraic operations' )

keys = word_embeddings.keys ()
data = []
for key in keys: data.append ( word_embeddings [ key ] )

embedding = pd.DataFrame ( data = data , index = keys )

print ( 'Define a function to find the closest word to a vector' )

def find_closest_word ( v , k = 1 ) :
    diff = embedding.values - v
    
    # Get the norm of each difference vector.
    # norm is the distance from (0,0).
    delta = np.sum ( diff * diff , axis=1 )
    
    # Find the index of the minimun distance in the array
    i = np.argmin ( delta )
    
    return embedding . iloc [ i ] . name


print ( 'Print some rows of the embedding as a Dataframe' )

embedding.head ( 10 )

find_closest_word ( np.array ( country ) )

print ( '' )
# ---------------------------------------------------------------------------------------------

print ( 'Predicting other Countries' )

find_closest_word ( np.array ( vec ( 'Italy' ) ) - np.array ( vec ( 'Rome' ) ) + np.array ( vec ( 'Madrid' ) ) )

print ( find_closest_word ( np.array ( vec ( 'Berlin'  ) ) + capital ) )
print ( find_closest_word ( np.array ( vec ( 'Beijing' ) ) + capital ) )
print ( find_closest_word ( np.array ( vec ( 'Lisbon'  ) ) + capital ) )

print ( '' )
# ---------------------------------------------------------------------------------------------


print ( 'Represent a sentence as a vector' )

doc = "Spain petroleum city king"
vdoc = [ np.array ( vec ( x ) ) for x in doc.split ( " " ) ]
doc2vec = np.sum ( vdoc , axis = 0 )

print ( find_closest_word ( doc2vec ) )

