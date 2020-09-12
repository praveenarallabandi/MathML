import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def calculateDerivate(m, x, y, w0, w1, w2, w3, w4, w5):
        print('calculateDerivate : ', x, y)
        dw_0 = 1.0 / m * sum([(2 * (w0 - y[i])) for i in range(m)])
        dw_1 = 1.0 / m * sum([(2 * (w0 + (w1 * x[i]) - y[i])) * x[i] for i in range(m)])
        dw_2 = 1.0 / m * sum([(2 * (w0 + (w1 * x[i]) + (w2 * x[i] ** 2)) - y[i]) * (x[i] ** 2) for i in range(m)])
        dw_3 = 1.0 / m * sum([(2 * (w0 + (w1 * x[i]) + (w2 * x[i] ** 2) + (w3 * x[i] ** 3)) - y[i]) * (x[i] ** 3) for i in range(m)])
        dw_4 = 1.0 / m * sum([(2 * (w0 + (w1 * x[i]) + (w2 * x[i] ** 2) + (w3 * x[i] ** 3) + (w4 * x[i] ** 4)) - y[i]) * (x[i] ** 4) for i in range(m)])
        dw_5 = 1.0 / m * sum([(2 * (w0 + (w1 * x[i]) + (w2 * x[i] ** 2) + (w3 * x[i] ** 3) + (w4 * x[i] ** 4) + (w5 * x[i] ** 5)) - y[i]) * (x[i] ** 5) for i in range(m)])
        print('calculateDerivate Result : dw_0 : {}, dw_1: {}, dw_2: {}. dw_3: {}, dw_4: {}. dw_5: {}'.format(dw_0, dw_1, dw_2, dw_3, dw_4, dw_5))
        return dw_0,dw_1,dw_2,dw_3,dw_4,dw_5

def calculateNewDerivate(w, grad, alpha):
        w_new = w - alpha * grad
        return w_new

def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
        converged = False
        iter = 0
        m = x.shape[0] # number of samples
        print('value m : {}'.format(m))
        # initial w_n values
        w0 = np.random.rand(m)
        w1 = np.random.rand(m)
        w2 = np.random.rand(m)
        w3 = np.random.rand(m)
        w4 = np.random.rand(m)
        w5 = np.random.rand(m)

        # total error, J(theta)(w-good)
        J = sum([(w0 + w1 + w2 + w3+ w4 + w5 * x[i] - y[i]) ** 2 for i in range(m)])

        # Iterate Loop
        while not converged:
                # for each training sample, compute the gradient (d/d_theta j(theta))
                # grad0 = 1.0 / m * sum([(t0 + t1 * x[i] - y[i]) for i in range(m)]) 
                # grad1 = 1.0 / m * sum([(t0 + t1 * x[i] - y[i]) * x[i] for i in range(m)])
                grad0, grad1, grad2, grad3, grad4, grad5 = calculateDerivate(m, x, y, w0, w1, w2, w3, w4, w5)

                # update the theta_temp
                w0_new = calculateNewDerivate(w0, grad0, alpha)
                w1_new = calculateNewDerivate(w1, grad1, alpha)
                w2_new = calculateNewDerivate(w2, grad2, alpha) 
                w3_new = calculateNewDerivate(w3, grad3, alpha)
                w4_new = calculateNewDerivate(w4, grad4, alpha)
                w5_new = calculateNewDerivate(w5, grad5, alpha)
        
                # update w
                w0 = w0_new
                w1 = w1_new
                w2 = w2_new
                w3 = w3_new
                w4 = w4_new
                w5 = w5_new

                # mean squared error
                e = sum( [ (w0 + w1 + w2 + w3 + w4 + w5 * x[i] - y[i]) ** 2 for i in range(m)] ) 
                print('mean :ep:{}, abs{}'.format(ep, abs(J-e)))
                
                if abs(J-e).all() <= ep:
                        print('Converged, iterations: ', iter, '!!!')
                        converged = True
        
                J = e   # update error 
                iter += 1  # update iter
        
                if iter == max_iter:
                        print ('Max interactions exceeded!')
                        converged = True

        return w0,w1,w2,w3,w4,w5

if __name__ == '__main__':
        # x, y = make_regression(n_samples=100, n_features=1, n_informative=1, random_state=0, noise=35) 
        # print ('x.shape = %s y.shape = %s') %(x.shape, y.shape)
        print('Main START')
        x = np.array([-1.67245526, -2.36540279, -2.14724263,  1.40539096,  1.24297767,
                -1.71043904,  2.31579097,  2.40479939, -2.22112823])

        y = np.array([-18.56122168, -24.99658931, -24.41907817,  -2.688209,
                -1.54725306,  -19.18190097,   1.74117419,
                3.97703338, -24.80977847])
        print('x.shape = {} y.shape = {}'.format(x.shape, y.shape))
        plt.scatter(x, y)
        # plt.show()
        
        alpha = 0.01 # learning rate
        ep = 0.01 # convergence criteria

        # call gredient decent, and get intercept(=theta0) and slope(=theta1)
        theta0, theta1, theta2, theta3, theta4, theta5 = gradient_descent(alpha, x, y, ep, max_iter=1000)
        # print ('theta0 = %s theta1 = %s') % (theta0, theta1) 
        print('theta0 = {} theta1 = {} theta2 = {} theta3 = {} theta4 = {} theta5 = {}'.format(theta0, theta1, theta2, theta3, theta4, theta5))

                # check with scipy linear regression 
        # slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x[:,0], y)
        # print ('intercept = %s slope = %s') %(intercept, slope) 
        # print('intercept = {} slope = {}'.format(intercept, slope))

                # plot
        m = x.shape[0]
        for i in range(m):
                y_predict = (theta0 + theta1 + theta2 + theta3 + theta4 + theta5) * x 

        plt.plot(x,y,'o')
        plt.plot(x,y_predict,'k-')
        plt.show()
        print("Done!")
