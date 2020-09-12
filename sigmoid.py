import numpy as np 

def sigmoid(sop):
    return 1.0 / (1 + np.exp(-1 * sop))


def error(predicted, target):
    return np.power(predicted - target, 2)


def error_predicted_deriv(predicted, target):
    return 2 * (predicted - target)


def sigmoid_sop_deriv(sop):
    return sigmoid(sop) * (1.0 - sigmoid(sop))


def sop_w_deriv(x):
    return x


def update_w(w, grad, learning_rate):
    return w - learning_rate * grad


x = np.array([0.1, 0.4, 4.1, 4.3, 1.8, 2.0, 0.01, 0.9, 3.8, 1.6])
target = np.array([0.2])

learning_rate = 0.001

w1_10 = np.random.rand(10)
w2_10 = np.random.rand(10)
w3_2 = np.random.rand(2)
w3_2_old = w3_2
print("Initial W : ", w1_10, w2_10, w3_2)

for k in range(80000):
    # Forward Pass
    # Hidden Layer Calculations
    sop1 = np.sum(w1_10 * x)
    sop2 = np.sum(w2_10 * x)

    sig1 = sigmoid(sop1)
    sig2 = sigmoid(sop2)

    # Output Layer Calculations
    sop3 = np.sum(w3_2 * np.array([sig1, sig2]))

    predicted = sigmoid(sop3)
    err = error(predicted, target)

    # Backward Pass
    g1 = error_predicted_deriv(predicted, target)

    ### Working with weights between hidden and output layer
    g2 = sigmoid_sop_deriv(sop3)

    g3 = np.zeros(w3_2.shape[0])
    g3[0] = sop_w_deriv(sig1)
    g3[1] = sop_w_deriv(sig2)

    grad_hidden_output = g3 * g2 * g1

    w3_2[0] = update_w(w3_2[0], grad_hidden_output[0], learning_rate)
    w3_2[1] = update_w(w3_2[1], grad_hidden_output[1], learning_rate)

    ### Working with weights between input and hidden layer
    # First Hidden Neuron
    g3 = sop_w_deriv(w3_2_old[0])
    g4 = sigmoid_sop_deriv(sop1)

    g5 = np.zeros(w1_10.shape[0])
    g5 = sop_w_deriv(x)

    grad_hidden1_input = g5 * g4 * g3 * g2 * g1

    w1_10 = update_w(w1_10, grad_hidden1_input, learning_rate)

    # Second Hidden Neuron
    g3 = sop_w_deriv(w3_2_old[1])
    g4 = sigmoid_sop_deriv(sop2)

    g5 = np.zeros(w2_10.shape[0])
    g5 = sop_w_deriv(x)

    grad_hidden2_input = g5 * g4 * g3 * g2 * g1

    w2_10 = update_w(w2_10, grad_hidden2_input, learning_rate)

    w3_2_old = w3_2
    print(predicted)
