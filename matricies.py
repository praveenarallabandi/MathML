import sys
import numpy as np

print('Python: {}'.format(sys.version))
print('NumPy: {}'.format(np.__version__))

# defining a scalar
x = 6
print(x)

# defining a vector
x = np.array((1,2,3))
print(x)
print('Vector Dimensions: {}'.format(x.shape))
print('Vector Size: {}'.format(x.size))

# defining a matrix
x = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
print(x)
print('Matrix Dimensions: {}'.format(x.shape))
print('Matrix Size: {}'.format(x.size))

# define matrix of given dimension
x = np.ones((5,5))
print(x)

# Tensor dimension
x = np.ones((3,3,3))
print(x)
print('Tensor Dimensions: {}'.format(x.shape))
print('Tensor Size: {}'.format(x.size))

# indexing
A = np.ones((5,5), dtype = np.int)
print(A)
A[0,1] = 2
print(A)

A[:,0] = 3
print(A)
A[:,:] = 5
print(A)

# for higher demnsions simply add an index
A = np.ones((5,5,5), dtype = np.int)
A[:,0,0] = 6
print(A)

# define matrix operations
A = np.matrix([[1,2],[3,4]])
B = np.ones((2,2), dtype = np.int)
print(A)
print(B)
# Element wise sum
C = A + B
print('Add: {}',C)
# Element wise sub
C = A - B
print('Sub: {}',C)
# Matrix multiplication
C = A * B
print('Mul: {}',C)
# Matrix transpose
A = np.array(range(9))
A = A.reshape(3,3)
print(A)
print('Transpose:',A.T)

A = np.array(range(10))
A = A.reshape(2,5)
print(A)
print('Transpose:2by5:',A.T)

# Tensors
T = np.ones((5,5,5,5,5,5,5,5,5,5))
print('Tensor Dimensions: {}'.format(T.shape))
print('Tensor Length: {}'.format(len(T.shape)))
print('Tensor Size: {}'.format(T.size))