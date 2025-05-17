import numpy as np

LEARNING_RATE = 0.00001
EPOCHES = 1000


# neuron definition
def forward(x, w, b):
    return x.dot(w.T) + b


def mse_loss(p, y):
    return ((p - y) ** 2).mean()


def gradient(p, y):
    return (p - y) * 2 / np.prod(y.shape)


def backward(x, d, w, b):
    w -= d * x * LEARNING_RATE
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
feature_shape = features.shape[-1], labels.shape[-1]

weight = np.ones((feature_shape[1], feature_shape[0]))
bias = np.zeros(feature_shape[1])

# epoch
for epoch in range(EPOCHES):
    print(f"Epoch: {epoch}")

    # iteration
    for i in range(len(features)):
        feature = features[i]
        label = labels[i]

        # prediction
        prediction = forward(feature, weight, bias)
        print(f'Prediction: {prediction}')

        # evaluation
        error = mse_loss(prediction, label)
        print(f'Error: {error}')

        # backpropagation
        delta = gradient(prediction, label)
        (weight, bias) = backward(feature, delta, weight, bias)
        print(f"New weight: {weight}")
        print(f"New bias: {bias}")
