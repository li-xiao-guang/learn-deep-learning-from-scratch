import numpy as np

LEARNING_RATE = 0.00001
EPOCHES = 1000
BATCHES = 2


# neuron definition
def forward(x, w, b):
    return x.dot(w.T) + b


# loss function
def mse_loss(p, y):
    return ((p - y) ** 2).mean()


# backpropagation
def gradient(p, y):
    return (p - y) * 2 / np.prod(y.shape)


def backward(x, d, w, b):
    w -= d.T.dot(x) * LEARNING_RATE
    b -= np.sum(d, axis=0) * LEARNING_RATE
    return w, b


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
weight = np.ones((labels.shape[-1], features.shape[-1]))
bias = np.zeros(labels.shape[-1])

# epoch
for epoch in range(EPOCHES):
    print(f"Epoch: {epoch}")

    # iteration
    for i in range(0, len(features), BATCHES):
        feature = features[i: i + BATCHES]
        label = labels[i: i + BATCHES]

        # prediction
        prediction = forward(feature, weight, bias)

        # evaluation
        error = mse_loss(prediction, label)

        # backpropagation
        delta = gradient(prediction, label)
        (weight, bias) = backward(feature, delta, weight, bias)

    print(f'Prediction: {prediction}')
    print(f'Error: {error}')
    print(f"New weight: {weight}")
    print(f"New bias: {bias}")
