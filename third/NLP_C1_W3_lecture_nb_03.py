#!/usr/bin/env python

import numpy as np                         # Linear algebra library
import matplotlib.pyplot as plt            # library for visualization
from sklearn.decomposition import PCA      # PCA library
import pandas as pd                        # Data frame library
import math                                # Library for math functions
import random                              # Library for pseudo random numbers

import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms



def savePictures( fig , name ) : return fig.savefig ( '{}.png'.format( name ) )
print ( '' ) ; # ----------------------------------------------------------------------



x , y = None , None
data = None ; rotationMatrix = None
std1 , std2 = None , None

def MAKE_DATA1_INIT1 () :

    global x , y
    print ( 'The amount of the correlation' ) ; n = 1
    print ( 'Generate 1000 samples from a uniform random variable' ) ; x = np.random.uniform ( 1 , 2 , 1000 )
    print ( 'Make y = n * x' ) ; y = x.copy() * n
    print ( 'PCA works better if the data is centered' )
    print ( 'Center x. Remove its mean' ) ; x = x - np.mean(x)
    print ( 'Center y. Remove its mean' ) ; y = y - np.mean(y)
      
MAKE_DATA1_INIT1 ()
print ( '' ) ; # ----------------------------------------------------------------------



def USING_DATA1_PCA () :

    fig , ax = plt.subplots () ; global data

    print ( 'Create a data frame with x and y' ) ; data = pd.DataFrame({'x': x, 'y': y})
    print ( 'Plot the original correlated data in blue' ) ; plt.scatter(data.x, data.y)

    print ( 'Instantiate a PCA. Choose to get 2 output variables' ) ; pca   = PCA(n_components=2)
    print ( 'Create the transformation model for this data.' )      ; pcaTr = pca.fit(data)

    print ( 'Transform the data base on the rotation matrix of pcaTr' ) ; rotatedData = pcaTr.transform(data)
    print ( 'Create a data frame with the new variables.' ) ; dataPCA = pd.DataFrame(data = rotatedData, columns = ['PC1', 'PC2'])

    print ( 'Plot the transformed data in orange' ) ; plt.scatter(dataPCA.PC1, dataPCA.PC2)

    print ( 'Understanding the transformation model pcaTr' )
    print('Eigenvectors or principal component: First row must be in the direction of [1, n]') ; print ( pcaTr.components_ )
    print('Eigenvalues or explained variance') ; print ( pcaTr.explained_variance_ )

    savePictures( fig , 'PCA_data_normal_transformed_1' ) # plt.show () # exit ()
    
USING_DATA1_PCA ()
print ( '' ) ; # ----------------------------------------------------------------------



def MAKE_DATA2_INIT2 () :

    global data , std1 , std2 , rotationMatrix ; random.seed ( 100 )
    
    print ( 'The desired standard deviation of our  first random variable' ) ; std1 = 1
    print ( 'The desired standard deviation of our second random variable' ) ; std2 = 0.333
    print ( 'Get 1000 samples from x ~ N(0, std1)' ) ; x = np.random.normal(0, std1, 1000)
    print ( 'Get 1000 samples from y ~ N(0, std2)' ) ; y = np.random.normal(0, std2, 1000)

    # y = y + np.random.normal (0,1,1000) * noiseLevel * np.sin(0.78)

    print ( 'PCA works better if the data is centered' )
    print ( 'Center x' ) ; x = x - np.mean(x) 
    print ( 'Center y' ) ; y = y - np.mean(y)

    print ( 'Define a pair of dependent variables with a desired amount of covariance' )
    print ( 'Magnitude of covariance' ) ; n = 1 
    print ( 'Convert the covariance to and angle' ) ; angle = np.arctan(1 / n)
    print ( 'angle: ',  angle * 180 / math.pi )

    print ( 'Create a rotation matrix using the given angle' )
    rotationMatrix = np.array([[+np.cos(angle), np.sin(angle)],
                               [-np.sin(angle), np.cos(angle)]])

    print('rotationMatrix') ; print ( rotationMatrix )

    print ( 'Create a matrix with columns x and y' ) ; xy = np.concatenate(([x] , [y]), axis=0).T
    print ( 'Transform the data using the rotation matrix. It correlates the two variables' )
    print ( 'Return a nD array' ) ; data = np.dot(xy, rotationMatrix)
    
MAKE_DATA2_INIT2 ()
print ( '' ) ; # ----------------------------------------------------------------------



def USING_DATA2_PART1_PLOT () :

    global data

    fig , ax = plt.subplots ()
    print ( 'Print the rotated data' ) ; plt.scatter(data[:,0], data[:,1])
    savePictures( fig , 'Rotated_data' ) # plt.show() # exit ()

USING_DATA2_PART1_PLOT ()
print ( '' ) ; # ----------------------------------------------------------------------



def USING_DATA2_PART2_PCA () :

    global data , std1 , std2 , rotationMatrix ; fig , ax = plt.subplots ()
    
    print ( 'Print the original data in blue' ) ; plt.scatter(data[:,0], data[:,1])

    print ( 'Using the rotated data' )
    print ( 'Instantiate a PCA. Choose to get 2 output variables' ) ; pca   = PCA(n_components=2)
    print ( 'Create the transformation model for this data.' ) ;      pcaTr = pca.fit(data)
    print ( 'Transform the data base on the rotation matrix of pcaTr' ) ; dataPCA = pcaTr.transform(data)

    print('Eigenvectors or principal component: First row must be in the direction of [1, n]') ; print(pcaTr.components_)
    print('Eigenvalues or explained variance') ;                                                 print(pcaTr.explained_variance_)

    print ( 'Print the rotated data' ) ; plt.scatter(dataPCA[:,0], dataPCA[:,1])
    print ( 'Plot the first component axe. Use the explained variance to scale the vector' )
    plt.plot([0, rotationMatrix[0][0] * std1 * 3], [0, rotationMatrix[0][1] * std1 * 3], 'k-', color='red')
    print ( 'Plot the second component axe. Use the explained variance to scale the vector' )
    plt.plot([0, rotationMatrix[1][0] * std2 * 3], [0, rotationMatrix[1][1] * std2 * 3], 'k-', color='green')

    savePictures ( fig , 'PCA_data_normal_transformed_2' ) # plt.show() # exit ()
    
USING_DATA2_PART2_PCA ()
print ( '' ) ; # ----------------------------------------------------------------------



def USING_ALL_DATA_PLOT () :

    global data ; fig , ax = plt.subplots () ; nPoints = len(data)

    print ( 'Plot the original data in blue' ) ; plt.scatter(data[:,0], data[:,1])
    print ( 'Plot the projection along the first component in orange' ) ; plt.scatter(data[:,0], np.zeros(nPoints))
    print ( 'Plot the projection along the second component in green' ) ; plt.scatter(np.zeros(nPoints), data[:,1])

    savePictures ( fig , 'PCA_data_normal_transformed_3' ) # plt.show()
    
USING_ALL_DATA_PLOT ()
