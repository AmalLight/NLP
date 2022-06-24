#!/usr/bin/env python

import numpy as np                # library for array and matrix manipulation
import matplotlib.pyplot as plt   # visualization library
#from utils_nb import plot_vectors # helper function to plot vectors
import inspect
import pprint                     # utilities for console printing 

# ------------------------------------------------

# print ( inspect.getsource ( plot_vectors ) )

def plot_vectors(vectors, colors=['k', 'b', 'r', 'm', 'c'], axes=None, fname='image.svg', ax=None):
    scale = 1
    scale_units = 'x'
    x_dir = []
    y_dir = []
    
    for i, vec in enumerate(vectors):
        x_dir.append(vec[0][0])
        y_dir.append(vec[0][1])
    
    if ax == None:
        fig, ax2 = plt.subplots()
    else:
        ax2 = ax
      
    if axes == None:
        x_axis = 2 + np.max(np.abs(x_dir))
        y_axis = 2 + np.max(np.abs(y_dir))
    else:
        x_axis = axes[0]
        y_axis = axes[1]
        
    ax2.axis([-x_axis, x_axis, -y_axis, y_axis])
        
    for i, vec in enumerate(vectors):
        ax2.arrow(0, 0, vec[0][0], vec[0][1], head_width=0.05 * x_axis, head_length=0.05 * y_axis, fc=colors[i], ec=colors[i])
    
    if ax == None:
        #plt.show()
        fig.savefig(fname)
    
    print ( '' )
    print ( fname , 'saved with vectors:' , vectors , 'and axes:' , axes )
    print ( '' )

# ------------------------------------------------

print ( 'Instantiate a pretty printer' ) ; pp = pprint.PrettyPrinter(indent=4)

# ------------------------------------------------

def basic_hash_table(value_l, n_buckets):
    
    def hash_function(value, n_buckets): return int(value) % n_buckets
    
    print ( 'Initialize all the buckets in the hash table as empty lists' )
    
    hash_table = {i:[] for i in range(n_buckets)}

    for value in value_l:
    
        print ( 'Get the hash key for the given value' )
        hash_value = hash_function(value,n_buckets)
        
        print ( 'Add the element to the corresponding bucket' )
        hash_table[hash_value].append(value)
    
    return hash_table

# ------------------------------------------------

print ( 'Now let\'s see the hash table function in action. The pretty print function (`pprint()`) will produce a visually appealing output.' )

print ( 'Set of values to hash' ) ; value_l = [100, 10, 14, 17, 97] ; print ( 'value_l:' , value_l )

hash_table_example = basic_hash_table(value_l, n_buckets=10)

pp.pprint ( hash_table_example )

print ( '' )
# ------------------------------------------------

glob = {}

def gset ( var , value ) : glob [ var ] = value
def gget ( var ) : return  glob [ var ]

def printdefine ( string , var , value ) :

    gset ( var , value )
    
    print ( string , ',' , var , ':' , gget ( var ) )
    
    return gget ( var )

# ------------------------------------------------

printdefine ( 'Define a single plane' , 'P' , np.array ( [ [ 1, 1 ] ] ) )

print ( 'Create a plot' ) ; fig, ax1 = plt.subplots ( figsize=(8, 8) )

print ( 'Plot the plane P as a vector' ) ; plot_vectors([ gget ( 'P' ) ], axes=[2, 2], ax=ax1)

print ( 'Plot 10 random points.' )

for i in range(0, 10):
        printdefine ( 'Get a pair of random numbers between -4 and 4' , 'v1' , np.array(np.random.uniform(-2, 2, 2) ) )
        
        side_of_plane = np.sign(np.dot( gget ( 'P' ), gget ( 'v1' ).T))
        
        print ( 'Color the points depending on the sign of the result of np.dot(P, point.T)' )
        
        if side_of_plane == 1: print ( 'Plot blue points' ) ; ax1.plot([ gget ( 'v1' )[0]], [ gget ( 'v1' )[1]], 'bo')
        else:                  print ( 'Plot red points'  ) ; ax1.plot([ gget ( 'v1' )[0]], [ gget ( 'v1' )[1]], 'ro')

# plt.show()

fig.savefig ( 'image1_plot_dot_between_P_and_10_v1.png' )

print ( 'fig\'s png image1 saved correctly' )

print ( '' )
# ------------------------------------------------

printdefine ( 'Define a single plane. You may change the direction' , 'P' , np.array([[1, 2]]) )

printdefine ( 'Get a new plane perpendicular to P. We use a rotation matrix' , 'PT' , np.dot([[0, 1], [-1, 0]], gget ( 'P' ).T ).T  )

print ( 'Create a plot with custom size' ) ; fig, ax1 = plt.subplots(figsize=(8, 8))

print ( 'Plot the plane P as a vector' ) ; plot_vectors([ gget ( 'P' )], colors=['b'], axes=[2, 2], ax=ax1)

print ( 'Plot the plane P as a 2 vectors.' )

print ( 'We scale by 2 just to get the arrows outside the current box' )

plot_vectors([ gget ( 'PT' ) * 4, gget ( 'PT' ) * -4], colors=['k', 'k'], axes=[4, 4], ax=ax1)

print ( 'Plot 20 random points.' )

for i in range(0, 20):
        printdefine ( 'Get a pair of random numbers between -4 and 4' , 'v1' , np.array(np.random.uniform(-4, 4, 2)) )
        
        printdefine ( 'Get dot result', 'dotresult' , np.dot( gget ( 'P' ), gget ( 'v1' ).T) )
        
        printdefine ( 'Get the sign of the dot product with P' , 'sign' , np.sign( gget ( 'dotresult' ) ) )
        
        print ( 'Color the points depending on the sign of the result of np.dot(P, point.T)' )
        
        if gget ( 'sign' ) == 1: print ( 'Plot blue points' ) ; ax1.plot([ gget ( 'v1' )[0]], [ gget ( 'v1' )[1]], 'bo')
        else:                    print ( 'Plot red points'  ) ; ax1.plot([ gget ( 'v1' )[0]], [ gget ( 'v1' )[1]], 'ro')

fig.savefig ( 'image2_plot_dot_between_P_and_20_v1.png' )

print ( 'fig\'s png image2 saved correctly' )

print ( '' )
# ------------------------------------------------

print ( 'Now, let us see what is inside the code that color the points.' )

printdefine ( 'Single plane' , 'P' , np.array( [[1, 1]] ) )

printdefine ( 'Sample point 1', 'v1' , np.array( [[1, 2]] ) )

printdefine ( 'Sample point 2', 'v2' , np.array( [[-1, 1]] ) )

printdefine ( 'Sample point 3', 'v3' , np.array( [[-2, -1]] ) )

np.dot( gget ( 'P' ), gget ( 'v1' ).T )

np.dot( gget ( 'P' ), gget ( 'v2' ).T )

np.dot( gget ( 'P' ), gget ( 'v3' ).T )

def side_of_plane(P, v):

    printdefine ( 'Get the dot product P * v' , 'dotproduct' , np.dot(P, v.T) )
    
    printdefine ( 'The sign of the elements of the dotproduct matrix' , 'sign_of_dot_product' , np.sign( gget ( 'dotproduct' ) ) )
    
    printdefine ( 'The value of the first item' , 'sign_of_dot_product_scalar' , gget ( 'sign_of_dot_product' ).item() )
    
    return gget ( 'sign_of_dot_product_scalar' )
 
side_of_plane( gget ( 'P' ) , gget ( 'v1' ) )
side_of_plane( gget ( 'P' ) , gget ( 'v2' ) )
side_of_plane( gget ( 'P' ) , gget ( 'v3' ) )

print ( '' )
# ------------------------------------------------

printdefine ( '' , 'P1' , np.array([[1, 1]]) )

printdefine ( '' , 'P2' , np.array([[-1, 1]]) )

printdefine ( '' , 'P3' , np.array([[-1, -1]]) )

printdefine ( '' , 'P_l' , [ gget ( 'P1' ) , gget ( 'P2' ) , gget ( 'P3' ) ] )

printdefine ( 'Vector to search' , 'v' , np.array([[2, 2]]) )

def hash_multi_plane(P_l, v):

    hash_value = 0
    
    for i, P in enumerate(P_l):
    
        sign = side_of_plane(P,v)
        hash_i = 1 if sign >=0 else 0
        hash_value += 2**i * hash_i
    
    return hash_value

printdefine ( 'Find the number of the plane that containes this value' , 'hash_multi_plane_1' ,  hash_multi_plane ( gget ( 'P_l' ), gget ( 'v' ) ) )

print ( '' )
# ------------------------------------------------

np.random.seed(0)

printdefine ( '' , 'num_dimensions' , 2 )
printdefine ( '' , 'num_planes' , 3 )
printdefine ( '' , 'random_planes_matrix' , np.random.normal( size=( gget ( 'num_planes' ), gget ( 'num_dimensions' )) ) )

printdefine ( 'Vector to search' , 'v' , np.array([[2, 2]]) )

print ( 'The next function is similar to the `side_of_plane()` function, but it evaluates more than a plane each time.' )
print ( 'The result is an array with the side of the plane of `v`, for the set of planes `P`' )

def side_of_plane_matrix(P, v):

    print ( 'Side of the plane function. The result is a matrix' )
    
    printdefine ( 'Sniff for dot result' , 'dotproduct' , np.dot(P, v.T) )
    printdefine ( 'Sniff for sign result' , 'sign' , np.sign( gget ( 'dotproduct' ) ) )
    
    return gget ( 'sign' )

printdefine ( '', 'sides_l' , side_of_plane_matrix( gget ( 'random_planes_matrix' ), gget ( 'v' ) ) )

print ( '' )
# ------------------------------------------------

def hash_multi_plane_matrix(P, v, num_planes):

    sides_matrix = side_of_plane_matrix(P, v)
    
    hash_value = 0
    
    for i in range(num_planes):
    
        sign = sides_matrix[i].item()
        hash_i = 1 if sign >=0 else 0
        hash_value += 2**i * hash_i
        
    return hash_value

print ( 'Print the bucket hash for the vector `v = [2, 2]`.' )

hash_multi_plane_matrix ( gget ( 'random_planes_matrix' ), gget ( 'v' ), gget ( 'num_planes' ) )

print ( '' )
# ------------------------------------------------

# FIND BUG { pippo = { pluto = paperino } }
# printdefine ( 'Assign 3D vectors to Words' , ' word_embedding' ,
# {
#    "I": np.array([1,0,1]),
#    "love": np.array([-1,0,1]),
#    "learning": np.array([1,0,1])
# })

print ( 'Assign 3D vectors to Words , word_embedding: ' , end = '' ) ; word_embedding = {

    "I": np.array([1,0,1]),
    "love": np.array([-1,0,1]),
    "learning": np.array([1,0,1])
}

print ( word_embedding )

printdefine ( 'See for current Words' , 'words_in_document' ,
[
    'I',
    'love',
    'learning',
    'not_a_word'
])

printdefine ( 'Create a collect for sign for each embedding Words' , 'document_embedding', np.array([0,0,0]) )

for word in gget ( 'words_in_document' ) : gset ( 'document_embedding' , gget ( 'document_embedding' ) + word_embedding.get ( word , 0 ) )
    
print ( 'document_embedding:' , gget ( 'document_embedding' ) )

