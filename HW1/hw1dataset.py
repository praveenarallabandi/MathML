

import numpy as np
import matplotlib.pyplot

x = np.array([-1.67245526, -2.36540279, -2.14724263,  1.40539096,  1.24297767,
              -1.71043904,  2.31579097,  2.40479939, -2.22112823])

y = np.array([-18.56122168, -24.99658931, -24.41907817,  -2.688209,
              -1.54725306,  -19.18190097,   1.74117419,
              3.97703338, -24.80977847])

matplotlib.pyplot.scatter(x, y)


def gradient_descent(gamma, x, j, y, ep=0.0001, max_iter=10000):
    converged = False
    max_iter = 0
    m = x.shape[0]
    w0 = np.random.rand(x.shape[1])
    w1 = np.random.rand(x.shape[1])
    w2 = np.random.rand(x.shape[2])
    w3 = np.random.rand(x.shape[3])
    w4 = np.random.rand(x.shape[4])
    w5 = np.random.rand(x.shape[5])

    j = sum((w0 + w1 * x + w2 * x ** 2 + w3 * x **
             3 + w4 * x ** 4 + w5 * x ** 5) - y) ** 2


def gradient(dw_0, dw_1, dw_2, dw_3, dw_4, dw_5):
        m = x.shape[0]
          
        dw_0 = 1 / m * sum((w0 + w1 * x[i]) - y[i])
        dw_1 = 1 / m * sum((w0 + w1 * x[i]) - y[i]) * x[i]
        dw_2 = 1 / m * \
            sum((w0 + w1 * x[i] + w2 * x[i] ** 2) - y[i]) * (x[i] ** 2)
        dw_3 = 1 / m * \
            sum((w0 + w1 * x[i] + w2 * x[i] ** 2 +
                 w3 * x[i] ** 3) - y[i]) * (x[i] ** 3)
        dw_4 = 1 / m * \
            sum((w0 + w1 * x[i] + w2 * x[i] ** 2 + w3 * x[i]
                 ** 3 + w4 * x[i] ** 4) - y[i]) * (x[i] ** 4)
        dw_5 = 1 / m * sum((w0 + w1 * x[i] + w2 * x[i] ** 2 + w3 * x[i]
                            ** 3 + w4 * x[i] ** 4 + w5 * x[i] ** 5) - y[i]) * (x[i] ** 5)



w0_old = w0 - gamma * dw_0
w1_old = w1 - gamma * dw_1

w2_old = w2 - gamma * dw_2
w3_old = w3 - gamma * dw_3

w4_old = w4 - gamma * dw_4

w5_old = w5 - gamma * dw_5

w0 = w0_old
w1 = w1_old
w2 = w2_old
w3 = w3_old
w4 = w4_old
w5 = w5_old



# mean squared error
e = sum((w0 + w1 * x + w2 * x ** 2 + w3 * x **
         3 + w4 * x ** 4 + w5 * x ** 5) - y) ** 2


if abs(J-e) <= ep:
    print('Converged, iterations: ', iter, '!!!')
    converged = True

    J = e   # update error
    iter += 1  # update iter

    max_iter = 1000

if (iter == max_iter):
    print('Max interactions exceeded!')
    converged = True

gamma = 0.001  # learning rate
ep = 0.0001  # convergence criteria
return w0, w1, w2, w3, w4, w5


w0, w1, w2, w3, w4, w5 = gradient_descent(gamma, x, y, ep, max_iter=1000)
print('w0 = %s, w1 = %s, w2 = %s, w3 = %s, w4 = %s, w5 = %s', w0, w1, w2, w3, w4, w5)

slope, intercept, r_value, p_value, slope_std_error = stats.linregress(
    x[:, 0], y)
print('intercept = %s slope = %s') % (intercept, slope)

for i in range(x.shape[0]):
    y_predict = (w0 + w1 * x + w2 * x ** 2 + w3 *
                 x ** 3 + w4 * x ** 4 + w5 * x ** 5)

plt.plot(x, y, 'o')
plt.plot(x, y_predict, 'k-')
plt.show()
print("Done!")
