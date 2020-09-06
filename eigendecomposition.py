import sys
import matplotlib
import numpy as np

from numpy import linalg

# define an array
A = np.arange(9) - 3
print(A)

# define reshape
B = A.reshape((3,3))
print(B)

# Euclidean L2 norm - default
print(np.linalg.norm(A))
print(np.linalg.norm(B))

#  the Frogenius norm is the L2 norm for matrix
print(np.linalg.norm(B, 'fro'))

# the max norm (P = infinity)
print(np.linalg.norm(A, np.inf))
print(np.linalg.norm(B, np.inf))

# Vector normalization - normalization to produce unit vector
norm = np.linalg.norm(A)
A_unit = (A / norm)
print('Unit Normalization : ', A_unit)
# the magnitude of a unit vecotr is equal to 1
print('Unit Normalization - Normalization: ', np.linalg.norm(A_unit))

#find eigenvalues and eigenvectors for simple matrix
A = np.diag(np.arange(0,4))
print(A)
eigenvalues, eigenvectors = np.linalg.eig(A)
print('Eigenvalues :', eigenvalues)
print('Eigenvectors :', eigenvectors)

# the eigenvalue corresponds to eigenvecotor
print('Eigenvalues : {}'.format(eigenvalues[1]))
print('Eigenvectors : {}'.format(eigenvectors[:, 1]))

# verify eigen decomposition
matrix = np.matmul(np.diag(eigenvalues), np.linalg.inv(eigenvectors))
output = np.matmul(eigenvectors, matrix).astype(np.int)
print(output)

# import matlibplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

origin = [0,0,0]
fig = plt.figure(figsize=(18,10))
ax1 = fig.add_subplot(121, projection = '3d')
ax1.quiver(origin, origin, origin, eigenvectors[0, :], eigenvectors[1, :], eigenvectors[2, :], color = 'k')
ax1.set_xlim([-3,3])
ax1.set_ylim([-3,3])
ax1.set_zlim([-3,3])
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')
ax1.view_init(15, 30)
ax1.set_title('Before Multiplication')
plt.show()