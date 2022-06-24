#!/usr/bin/env python

import numpy as np                     # Import numpy for array manipulation
import matplotlib.pyplot as plt        # Import matplotlib for charts
#from utils_nb import plot_vectors     # Function to plot vectors (arrows)
import inspect

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

glob = {}

def returnassign ( var , value ) :

    global glob
    
    glob [ var ] = value
        
    return glob [ var ]

# ------------------------------------------------

print ( 'Create a 1 x 2 matrix x:' , returnassign ( 'x' , np.array([[1, 1]]) ) )

print ( 'Create a 2 x 2 matrix ro: {}'.format ( returnassign ( 'r' , np.array([[2, 0],[0, -2]]) ) ) )

print ( 'Apply the dot product between x and ro:' , returnassign ( 'y' , np.dot( glob [ 'x' ], glob [ 'r' ] )))

plot_vectors([ glob [ 'x' ] ], axes=[4, 4], fname='transform_x.svg')

plot_vectors([ glob [ 'x' ], glob [ 'y' ]], axes=[4, 4], fname='transformx_and_y.svg')

print ( 'Note that the output vector `y` (blue) is transformed in another vector.' )

print ( '' )
# ------------------------------------------------

print ( 'remember that for matrix a list [...] of matrix meaning one row' )
print ( 'so you have to extract the correct column values from [a,b,c] => a , b , c' )
print ( 'on default the python\'s matrix following the row bind rules' )

print ( '' )
# ------------------------------------------------

print ( 'convert degrees to radians:' , returnassign ( 'a' , 100 * (np.pi / 180) ) )

print ( 'convert radians to matrix ro:' , returnassign ( 'ro' , np.array([[np.cos( glob [ 'a' ] ), -np.sin( glob [ 'a' ] )], [np.sin( glob [ 'a' ]), np.cos( glob [ 'a' ])]]) ) )

print ( 'make a row vector x2:' , returnassign ( 'x2' , np.array([2, 2]).reshape(1, -1) ) )

print ( 'make a rotation using x2 and ro = y2 = x2 * ro:', returnassign ( 'y2' , np.dot( glob [ 'x2' ] , glob [ 'ro' ] )))

plot_vectors([glob [ 'x2' ], glob [ 'y2' ]], fname='transform_02.svg')

print('\n x2 norm:', np.linalg.norm( glob [ 'x2' ]) )
print('\n y2 norm:', np.linalg.norm( glob [ 'y2' ]) )
print('\n ro norm:', np.linalg.norm( glob [ 'ro' ]) )

print ( 'the norms don\'t change' )

print ( '' )
# ------------------------------------------------

print ( 'A as x3:' , returnassign ( 'A' , np.array([[2, 2],[2, 2]]) ) )

print ( '`np.square()` is a way to square each element of a matrix. It must be equivalent to use the * operator in Numpy arrays.' )

print ( 'Asquare:' , returnassign ( 'Asquare' , np.square( glob [ 'A' ] )))

print ( 'A Frobenius Norm:' , returnassign ( 'AFrob' , np.sqrt(np.sum( glob [ 'Asquare' ] ))))

print ( 'compare sqrt square with np.norm:' , np.sqrt(np.sum( glob [ 'ro' ] * glob [ 'ro' ] )), '== ', np.linalg.norm( glob [ 'ro' ] ))

