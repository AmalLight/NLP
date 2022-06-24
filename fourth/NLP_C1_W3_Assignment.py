#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('capitals.txt', delimiter=' ')
data.columns = ['city1', 'country1', 'city2', 'country2']

# print first five elements in the DataFrame
data.head(5)


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------


word_embeddings = pickle.load(open("word_embeddings_subset.p", "rb"))
len(word_embeddings)  # there should be 243 words that will be used in this assignment

print("dimension: {}".format(word_embeddings['Spain'].shape[0]))

# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    dot = np.dot ( A , B )
    norma = np.linalg.norm ( A )
    normb = np.linalg.norm ( B )
    cos = dot / ( norma * normb )

    ### END CODE HERE ###
    return cos

# feel free to try different words
king = word_embeddings['king']
queen = word_embeddings['queen']

cosine_similarity(king, queen)


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------


# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def euclidean(A, B):
    """
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        d: numerical number representing the Euclidean distance between A and B.
    """

    # print ( type ( A ) )
    
    # print ( len ( A ) )
    
    # print ( A [ 0 ] )
    
    d = 0
    
    # for i in range ( len ( A ) ) : d += ( A [ i ] - B [ i ] ) ** 2
    
    d = np.linalg.norm ( A - B )
    
    # round ( np.sqrt ( d ) , 7 )

    return d

# Test your function
euclidean(king, queen)


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------


# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def get_country(city1, country1, city2, embeddings):
    """
    Input:
        city1: a string (the capital city of country1)
        country1: a string (the country of capital1)
        city2: a string (the capital city of country2)
        embeddings: a dictionary where the keys are words and values are their embeddings
    Output:
        countries: a dictionary with the most likely country and its similarity score
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # store the city1, country 1, and city 2 in a set called group
    group = set ( ( city1, country1, city2 ) )

    # get embeddings of city 1
    city1_emb = embeddings [ city1 ]

    # get embedding of country 1
    country1_emb = embeddings [ country1 ]

    # get embedding of city 2
    city2_emb = embeddings [ city2 ]

    # get embedding of country 2 (it's a combination of the embeddings of country 1, city 1 and city 2)
    # Remember: King - Man + Woman = Queen
    capital = city2_emb - city1_emb + country1_emb

    # Initialize the similarity to -1 (it will be replaced by a similarities that are closer to +1)
    similarity = -1

    # initialize country to an empty string
    country = ''

    # loop through all words in the embeddings dictionary
    for word in embeddings.keys():

        # first check that the word is not already in the 'group'
        if word not in group:

            # prediction for city2
            # city2_vec_emb = country2_vec_emb - country1_emb + city1_emb
            # country2_emb = embeddings [ word ]
            # city2_tmp_emb = country2_emb - country1_emb + city1_emb
            # cur_similarity = cosine_similarity ( city2_tmp_emb , city2_emb )
            
            country2_emb = embeddings [ word ]

            # calculate cosine similarity between embedding of country 2 and the word in the embeddings dictionary
            cur_similarity = cosine_similarity ( capital , country2_emb )

            # if the cosine similarity is more similar than the previously best similarity...
            if cur_similarity > similarity:

                # update the similarity to the new, better similarity
                similarity = cur_similarity

                # store the country as a tuple, which contains the word and the similarity
                country = (word,similarity)

    ### END CODE HERE ###

    return country

# Testing your function, note to make it more robust you can return the 5 most similar words.
get_country('Athens', 'Greece', 'Cairo', word_embeddings)


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------


# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def get_accuracy(word_embeddings, data):
    '''
    Input:
        word_embeddings: a dictionary where the key is a word and the value is its embedding
        data: a pandas dataframe containing all the country and capital city pairs
    
    Output:
        accuracy: the accuracy of the model
    '''

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # initialize num correct to zero
    num_correct = 0

    # loop through the rows of the dataframe
    for i in range ( len ( data ) ):

        # get city1
        #city1 = data [ 'city1' ] [ i ]
        city1 = data [ 'city1' ] [ i ]

        # get country1
        country1 = data [ 'country1' ] [ i ]

        # get city2
        city2 =  data [ 'city2' ] [ i ]

        # get country2
        y = country2 = data [ 'country2' ] [ i ]
        
        # TEST:
        # print ( country1 ) ; return 0

        # use get_country to find the predicted country2
        predicted_country2, _ = get_country(city1, country1, city2, word_embeddings)

        # if the predicted country2 is the same as the actual country2...
        if predicted_country2 == country2:
            # increment the number of correct by 1
            num_correct += 1

    # get the number of rows in the data dataframe (length of dataframe)
    m = len(data)

    # calculate the accuracy by dividing the number correct by m
    accuracy = num_correct / m

    ### END CODE HERE ###
    return accuracy

accuracy = get_accuracy(word_embeddings, data)
print(f"Accuracy is {accuracy:.2f}")


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------


# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def compute_pca(X, n_components=2):
    """
    Input:
        X: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output:
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    """

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    
    # mean center the data
    X2 = X_demeaned = X - np.mean ( X , axis=0 )
    
    # calculate the covariance matrix
    covariance_matrix = ( np.dot ( X2.T , X2 ) / ( len ( X2 ) - 1 ) ) #np.cov ( X_demeaned.T )
    #print ( covariance_matrix , covariance_matrix.shape ) # > Matrix 10,10

    # calculate eigenvectors & eigenvalues of the covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh ( covariance_matrix , UPLO='L' )

    # sort eigenvalue in increasing order (get the indices from the sort)
    idx_sorted = np.argsort ( eigen_vals )    
    #print ( idx_sorted , idx_sorted.shape ) # > 10 x 1
    
    # reverse the order so that it's from highest to lowest.
    idx_sorted_decreasing = np.flip ( idx_sorted )
    #print ( idx_sorted_decreasing , idx_sorted_decreasing.shape ) # > 10 x 1

    # sort the eigen values by idx_sorted_decreasing
    eigen_vals_sorted = np.array ( [ eigen_vals [ idx ] for idx in idx_sorted_decreasing ] )
    #print ( eigen_vals_sorted , eigen_vals_sorted.shape ) # > 10 x 1

    # sort eigenvectors using the idx_sorted_decreasing indices
    eigen_vecs_sorted = np.array ( [ eigen_vecs.T [ idx ] for idx in idx_sorted_decreasing ] ).T
    #print ( eigen_vecs_sorted , eigen_vecs_sorted.shape ) # > 10 x 10

    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    eigen_vecs_subset = eigen_vecs_sorted.T [ : n_components ] .T
    #print ( eigen_vecs_subset , eigen_vecs_subset.shape ) # > 10 x 2
    
    # transform the data by multiplying the transpose of the eigenvectors 
    # with the transpose of the de-meaned data
    # Then take the transpose of that product.
    X_reduced = np.dot ( eigen_vecs_subset.T , X_demeaned.T ).T
    
    ### END CODE HERE ###
    return X_reduced

# Testing your function
np.random.seed(1)
X = np.random.rand(3, 10)
X_reduced = compute_pca(X, n_components=2)
print("Your original matrix was " + str(X.shape) + " and it became:")
print(X_reduced)


# -----------------------------------------------------------------------
# -----------------------------------------------------------------------


words = ['oil', 'gas', 'happy', 'sad', 'city', 'town',
         'village', 'country', 'continent', 'petroleum', 'joyful']

# given a list of words and the embeddings, it returns a matrix with all the embeddings
X = get_vectors(word_embeddings, words)

print('You have 11 words each of 300 dimensions thus X.shape is:', X.shape)

# We have done the plotting for you. Just run this cell.
result = compute_pca(X, 2)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0] - 0.05, result[i, 1] + 0.1))

plt.show()
