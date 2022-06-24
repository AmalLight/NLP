#!/usr/bin/env python

import numpy  as np
import pandas as pd

import matplotlib.pyplot as plt
import utils
import inspect

from matplotlib.patches import Ellipse
from matplotlib         import transforms

# -----------------------------------------------------------------------------------

# print ( inspect.getsource ( confidence_ellipse ) )

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    
    print ( 'Using a special case to obtain the eigenvalues of this' )
    
    print ( 'two-dimensionl dataset.' )
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    print ( 'Calculating the stdandard deviation of x from' ,
            'the squareroot of the variance and multiplying' ,
            'with the given number of standard deviations.' )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
            
    print ( 'calculating the stdandard deviation of y ...' )
    
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    
    return ax.add_patch(ellipse)

# -----------------------------------------------------------------------------------

print ( 'Load the data from the csv file' )

data = pd.read_csv('bayes_features.csv')

# pd.set_option ( 'display.max_rows' , None )

# print ( data )

print ( 'Print the first 5 tweets features. Each row represents a tweet' )

print ( data.head(5) )

# -----------------------------------------------------------------------------------

print ( 'Plot the samples using columns 1 and 2 of the matrix' )

print ( 'reate a new figure with a custom size' )

fig, ax = plt.subplots(figsize = (8, 8))

print ( 'Define a color palete' )

colors = ['red', 'green']

print ( 'Color base on sentiment' )

print ( 'Plot a dot for each tweet' )

ax.scatter(data.positive, data.negative, c=[colors[int(k)] for k in data.sentiment], s = 0.1, marker='*')

print ( 'Custom limits for this chart' )

plt.xlim(-250,0)
plt.ylim(-250,0)

print ( 'x-axis label and y-axis label' )

plt.xlabel("Positive")
plt.ylabel("Negative")

fig.savefig ( 'data_positive_negative_1.png' )

# -----------------------------------------------------------------------------------

print ( 'Plot the samples using columns 1 and 2 of the matrix' )

fig, ax = plt.subplots(figsize = (8, 8))

print ( 'Define a color palete' )

colors = ['red', 'green']

print ( 'Color base on sentiment' )

print ( 'Plot a dot for tweet' )

ax.scatter(data.positive, data.negative, c=[colors[int(k)] for k in data.sentiment], s = 0.1, marker='*')

print ( 'Custom limits for this chart' )

plt.xlim(-200,40)  
plt.ylim(-200,40)

print ( 'x-axis label and y-axis label' )

plt.xlabel("Positive")
plt.ylabel("Negative")

print ( 'Filter only the positive samples' )

data_pos = data[data.sentiment == 1]

print ( 'Filter only the negative samples' )

data_neg = data[data.sentiment == 0]

print ( 'Print confidence ellipses of 2 std' )

confidence_ellipse(data_pos.positive, data_pos.negative, ax, n_std=2, edgecolor='black', label=r'$2\sigma$' )
confidence_ellipse(data_neg.positive, data_neg.negative, ax, n_std=2, edgecolor='orange')

print ( 'Print confidence ellipses of 3 std' )

confidence_ellipse(data_pos.positive, data_pos.negative, ax, n_std=3, edgecolor='black', linestyle=':', label=r'$3\sigma$')
confidence_ellipse(data_neg.positive, data_neg.negative, ax, n_std=3, edgecolor='orange', linestyle=':')
ax.legend()

# plt.show()

fig.savefig ( 'data_positive_negative_2.png' )

# -----------------------------------------------------------------------------------

print ( 'Copy the whole data frame' )

data2 = data.copy()

print ( 'The following 2 lines only modify the entries in the data frame where sentiment == 1' )

print ( 'Modify the negative attribute' )

data2.negative[data.sentiment == 1] =  data2.negative * 1.5 + 50

print ( 'Modify the positive attribute' )

data2.positive[data.sentiment == 1] =  data2.positive / 1.5 - 50

print ( 'Now let us plot the two distributions and the confidence ellipses' )

print ( 'Plot the samples using columns 1 and 2 of the matrix' )

fig, ax = plt.subplots(figsize = (8, 8))

print ( 'Define a color palete' )

colors = ['red', 'green']

print ( 'Color base on sentiment' )

# data.negative[data.sentiment == 1] =  data.negative * 2

print ( 'Plot a dot for tweet' )

ax.scatter(data2.positive, data2.negative, c=[colors[int(k)] for k in data2.sentiment], s = 0.1, marker='*')

print ( 'Custom limits for this chart' )

plt.xlim(-200,40)  
plt.ylim(-200,40)

print ( 'x-axis label and y-axis label' )

plt.xlabel("Positive")
plt.ylabel("Negative")

print ( 'Filter only the positive samples' )

data_pos = data2[data.sentiment == 1]

print ( 'Filter only the negative samples' )

data_neg = data[data2.sentiment == 0]

print ( 'Print confidence ellipses of 2 std' )

confidence_ellipse(data_pos.positive, data_pos.negative, ax, n_std=2, edgecolor='black', label=r'$2\sigma$' )
confidence_ellipse(data_neg.positive, data_neg.negative, ax, n_std=2, edgecolor='orange')

print ( 'Print confidence ellipses of 3 std' )

confidence_ellipse(data_pos.positive, data_pos.negative, ax, n_std=3, edgecolor='black', linestyle=':', label=r'$3\sigma$')
confidence_ellipse(data_neg.positive, data_neg.negative, ax, n_std=3, edgecolor='orange', linestyle=':')
ax.legend()

# plt.show()

fig.savefig ( 'data_positive_negative_3.png' )

print ( 'To give away: Understanding the data allows us to predict if the method will perform well or not.' )
print ( 'Alternatively, it will allow us to understand why it worked well or bad.' )

