#!/usr/bin/env python

import numpy as np

print ( 'Define a python list. It looks like an np array' )

alist = [1, 2, 3, 4, 5]

print ( 'Define a numpy array' )

narray = np.array([1, 2, 3, 4])

print(alist)
print(narray)

print(type(alist))
print(type(narray))

print ( '' )
# --------------------------------------------------------------------

print(narray + narray)
print(alist + alist)

print ( 'It is the same as with the product operator, `*`. In the first case, we scale the vector, while in the second case, we concatenate three times the same list.' )

print(narray * 3)
print(alist * 3)

print ( '' )
# --------------------------------------------------------------------

print ( '# Matrix initialized with NumPy arrays' )

npmatrix1 = np.array([narray, narray, narray])

print ( 'Matrix initialized with lists' )

npmatrix2 = np.array([alist, alist, alist])

print ( 'Matrix initialized with both types' )

npmatrix3 = np.array([narray, [1, 1, 1, 1], narray])

print(npmatrix1)
print(npmatrix2)
print(npmatrix3)

print ( '' )
# --------------------------------------------------------------------

print ( 'Define a 2x2 matrix' )

okmatrix = np.array([[1, 2], [3, 4]])

print ( 'Print okmatrix' )

print(okmatrix)

print ('Print a scaled version of okmatrix' )
 
print(okmatrix * 2)

print ( '' )
# --------------------------------------------------------------------

print ( 'Define a matrix. Note the third row contains 3 elements' )

badmatrix = np.array([[1, 2], [3, 4], [5, 6, 7]])

print ( 'Print the malformed matrix' )

print(badmatrix)

print ( 'It is supposed to scale the whole matrix' )

print(badmatrix * 2)

print ( '' )
# --------------------------------------------------------------------

print ( 'Scale by 2 and translate 1 unit the matrix' )

print ( 'For each element in the matrix, multiply by 2 and add 1' )

result = okmatrix * 2 + 1

print(result)

print ( '' )
# --------------------------------------------------------------------

print ( 'Add two sum compatible matrices' )

result1 = okmatrix + okmatrix
print(result1)

print ( 'Subtract two sum compatible matrices. This is called the difference vector' )

result2 = okmatrix - okmatrix
print(result2)

print ( '' )
# --------------------------------------------------------------------

print ( 'Multiply each element by itself' )

result = okmatrix * okmatrix # 
print(result)

print ( '' )
# --------------------------------------------------------------------

print ( 'Transpose a matrix' )

print ( 'Define a 3x2 matrix' )

matrix3x2 = np.array([[1, 2], [3, 4], [5, 6]])
print('Original matrix 3 x 2')
print(matrix3x2)
print('Transposed matrix 2 x 3')
print(matrix3x2.T)

print ( '' )
# --------------------------------------------------------------------

print ( 'Define an array' )

nparray = np.array([1, 2, 3, 4])
print('Original array')
print(nparray)
print('Transposed array')
print(nparray.T)

print ( '' )
# --------------------------------------------------------------------

print ( 'Define a 1 x 4 matrix. Note the 2 level of square brackets' )

nparray = np.array([[1, 2, 3, 4]])
print('Original array')
print(nparray)
print('Transposed array')
print(nparray.T)

print ( 'Calculating the norm of vector or even of a matrix is a general operation when dealing with data.' )
print ( 'Numpy has a set of functions for linear algebra in the subpackage **linalg**, including the **norm** function.' )
print ( 'Let us see how to get the norm a given array or matrix' )

print ( 'Define an array' )

nparray1 = np.array([1, 2, 3, 4])
norm1 = np.linalg.norm(nparray1)

print ( ' Define a 2 x 2 matrix. Note the 2 level of square brackets' )

nparray2 = np.array([[1, 2], [3, 4]])
norm2 = np.linalg.norm(nparray2) 

print(norm1)
print(norm2)

print ( '' )
# --------------------------------------------------------------------

print ( 'Note that without any other parameter, the norm function treats the matrix as being just an array of numbers.' )
print ( 'However, it is possible to get the norm by rows or by columns. The **axis** parameter controls the form of the operation: ' )
print ( '* **axis=0** means get the norm of each column * **axis=1** means get the norm of each row.' )

print ( 'Define a 3 x 2 matrix' )

nparray2 = np.array([[1, 1], [2, 2], [3, 3]]) 

print ( 'Get the norm for each column. Returns 2 elements' )

normByCols = np.linalg.norm(nparray2, axis=0)

print ( 'Get the norm for each row. Returns 3 elements' )

normByRows = np.linalg.norm(nparray2, axis=1)

print(normByCols)
print(normByRows)

print ( '' )
# --------------------------------------------------------------------

print ( 'However, there are more ways to get the norm of a matrix in Python.' )
print ( 'For that, let us see all the different ways of defining the dot product between 2 arrays.' )

print ( 'The dot product:' )

print ( 'Define an array' )

nparray1 = np.array([0, 1, 2, 3])

print ( 'Define an array' )

nparray2 = np.array([4, 5, 6, 7])

print ( 'Recommended way' )

flavor1 = np.dot(nparray1, nparray2)
print(flavor1)

print ( 'ok way' )

flavor2 = np.sum(nparray1 * nparray2)
print(flavor2)

print ( 'Geeks way' )

flavor3 = nparray1 @ nparray2
print(flavor3)

print ( 'As you never should do: Noobs way' )

flavor4 = 0
for a, b in zip(nparray1, nparray2):
    flavor4 += a * b
    
print(flavor4)

print ( '' )
# --------------------------------------------------------------------

print ( '**We strongly recommend using np.dot, since it is the only method that accepts arrays and lists without problems**' )

print ( 'Dot product on nparrays' )

norm1 = np.dot(np.array([1, 2]), np.array([3, 4]))

print ( 'Dot product on python lists' )

norm2 = np.dot([1, 2], [3, 4])

print(norm1, '=', norm2 )

print ( 'Finally, note that the norm is the square root of the dot product of the vector with itself. That gives many options to write that function:' )

print ( '' )
# --------------------------------------------------------------------

print ( '## Sums by rows or columns:' )

print ( 'Define a 3 x 2 matrix.' )

nparray2 = np.array([[1, -1], [2, -2], [3, -3]])

print ( 'Get the sum for each column. Returns 2 elements' )

sumByCols = np.sum(nparray2, axis=0)

print ( 'get the sum for each row. Returns 3 elements' )

sumByRows = np.sum(nparray2, axis=1)

print('Sum by columns: ')
print(sumByCols)
print('Sum by rows:')
print(sumByRows)

print ( '' )
# --------------------------------------------------------------------

print ( '## Get the mean by rows or columns:' )

print ( 'Define a 3 x 2 matrix. Chosen to be a matrix with 0 mean' )

nparray2 = np.array([[1, -1], [2, -2], [3, -3]])

print ( 'Get the mean for the whole matrix' )

mean = np.mean(nparray2)

print ( 'Get the mean for each column. Returns 2 elements' )

meanByCols = np.mean(nparray2, axis=0)

print ( 'get the mean for each row. Returns 3 elements' )

meanByRows = np.mean(nparray2, axis=1)

print('Matrix mean: ')
print(mean)
print('Mean by columns: ')
print(meanByCols)
print('Mean by rows:')
print(meanByRows)

print ( '## Center the columns of a matrix:' )

print ( 'Centering the attributes of a data matrix is another essential preprocessing step.' )
print ( 'Centering a matrix means to remove the column mean to each element inside the column.' )
print ( 'The sum by columns of a centered matrix is always 0.' )

print ( 'With NumPy, this process is as simple as this:' )

print ( 'Define a 3 x 2 matrix' )

nparray2 = np.array([[1, 1], [2, 2], [3, 3]])

print ( 'Remove the mean for each column' )

nparrayCentered = nparray2 - np.mean(nparray2, axis=0)

print('Original matrix')
print(nparray2)
print('Centered by columns matrix')
print(nparrayCentered)

print('New mean by column')
print(nparrayCentered.mean(axis=0))

print ( '**Warning:** This process does not apply for row centering. In such cases, consider transposing the matrix, centering by columns, and then transpose back the result.' )

print ( 'Define a 3 x 2 matrix' )

nparray2 = np.array([[1, 3], [2, 4], [3, 5]])

print ( 'Remove the mean for each row' )

nparrayCentered = nparray2.T - np.mean(nparray2, axis=1)

print ( 'Transpose back the result' )

nparrayCentered = nparrayCentered.T

print('Original matrix')
print(nparray2)
print('Centered by columns matrix')
print(nparrayCentered)

print('New mean by rows')
print(nparrayCentered.mean(axis=1))

print ( '' )
# --------------------------------------------------------------------

print ( 'Note that some operations can be performed using static functions like `np.sum()` or `np.mean()`, or by using the inner functions of the array' )

print ( 'Define a 3 x 2 matrix' )

nparray2 = np.array([[1, 3], [2, 4], [3, 5]])

print ( 'Static way' )

mean1 = np.mean(nparray2)

print ( 'Dinamic way' )

mean2 = nparray2.mean()

print(mean1, ' == ', mean2)

