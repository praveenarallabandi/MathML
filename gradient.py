import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print('matplotlib: {}'.format(matplotlib.__version__))
# generate 2D meshgrid
nx, ny = (100, 100)
x = np.linspace(0, 10, nx)
y = np.linspace(0, 10, ny)

xv, yv = np.meshgrid(x,y)
print('xv: ', xv)

# define a function to plot
def f(x,y):
    return x * (y**2)

# calculate Z value foreach x,y point
z = f(xv, yv)
print('z: ', z.shape)

# make color plots to display data
plt.figure(figsize=(14,12))
plt.pcolor(xv, yv, z)
plt.title('2D Color Plot of f(x,y) = xy^2')
plt.colorbar()
plt.show()

# generate 2D meshgrid for the gradient
nx, ny = (10,10)
x = np.linspace(0,10,nx)
y = np.linspace(0,10,ny)
xg, yg = np.meshgrid(x,y)

# calcualte the gradient of f(x,y)
Gy, Gx = np.gradient(f(xg, yg))
plt.figure(figsize=(14,12))
plt.pcolor(xv, yv, z)
plt.title('Gradient of f(x,y) = xy^2')
plt.colorbar()
plt.quiver(xg, yg, Gx, Gy, scale = 1000, color = 'w')
plt.show()

# calculate the gradient f(x,y) = xy^2 - Validation
def ddx(x,y):
    return y ** 2
def ddy(x,y):
    return 2 * x * y
Gx = ddx(xg, yg)
Gy = ddy(xg, yg)
plt.figure(figsize=(14,12))
plt.pcolor(xv, yv, z)
plt.title('Validate plot of [y^2, 2xy')
plt.colorbar()
plt.quiver(xg, yg, Gx, Gy, scale = 1000, color = 'w')
plt.show()
