import numpy as np

np.random.seed(99)

LEARNING_RATE = 0.00001
EPOCHES = 1000
BATCHES = 2


# neuron definition
def forward(x, w, b):
    return x.dot(w.T) + b


def mse_loss(p, y):
    return ((p - y) ** 2).mean()


def gradient(p, y):
    return (p - y) * 2 / np.prod(y.shape)


def gradient_backward(d, w):
    return d.dot(w)


def backward(x, d, w, b):
    w -= d.T.dot(x) * LEARNING_RATE
    b -= np.sum(d, axis=0) * LEARNING_RATE
    return w, b


def relu(x):
    return np.maximum(0, x)


def relu_backward(y, d):
    return (y > 0) * d


# dataset
features = np.array([[28.1, 58.0],
                     [22.5, 72.0],
                     [31.4, 45.0],
                     [19.8, 85.0],
                     [27.6, 63]])
labels = np.array([[165],
                   [95],
                   [210],
                   [70],
                   [155]])

# model
hidden_weight = np.random.rand(4, features.shape[-1]) / features.shape[-1]
hidden_bias = np.zeros(4)

weight = np.random.rand(labels.shape[-1], 4) / 4
bias = np.zeros(labels.shape[-1])

# epoch
for epoch in range(EPOCHES):
    print(f"Epoch: {epoch}")

    # iteration
    for i in range(0, len(features), BATCHES):
        feature = features[i: i + BATCHES]
        label = labels[i: i + BATCHES]

        # prediction
        hidden = relu(forward(feature, hidden_weight, hidden_bias))
        prediction = forward(hidden, weight, bias)

        # evaluation
        error = mse_loss(prediction, label)

        # backpropagation
        delta = gradient(prediction, label)
        hidden_delta = relu_backward(hidden, gradient_backward(delta, weight))

        (weight, bias) = backward(hidden, delta, weight, bias)
        (hidden_weight, hidden_bias) = backward(feature, hidden_delta, hidden_weight, hidden_bias)

    print(f'Prediction: {prediction}')
    print(f'Error: {error}')
    print(f"New weight: {weight}")
    print(f"New bias: {bias}")
    print(f"New hidden weight: {hidden_weight}")
    print(f"New hidden bias: {hidden_bias}")
