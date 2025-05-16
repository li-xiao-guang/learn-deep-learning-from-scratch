import numpy as np

np.random.seed(99)

LEARNING_RATE = 0.00001
EPOCHES = 1000
BATCH_SIZE = 2


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
in_feature_size = features.shape[-1]
out_feature_size = labels.shape[-1]
hidden_feature_size = 4
hidden_weight = np.random.rand(hidden_feature_size, in_feature_size) / in_feature_size
hidden_bias = np.zeros(hidden_feature_size)
weight = np.random.rand(out_feature_size, hidden_feature_size) / hidden_feature_size
bias = np.zeros(out_feature_size)

# epoch
for epoch in range(EPOCHES):
    print(f"Epoch: {epoch}")
    # iteration
    for i in range(0, len(features), BATCH_SIZE):
        feature = features[i: i + BATCH_SIZE]
        label = labels[i: i + BATCH_SIZE]

        # prediction
        hidden = forward(feature, hidden_weight, hidden_bias)
        prediction = forward(hidden, weight, bias)
        print(f'Prediction: {prediction}')

        # evaluation
        error = mse_loss(prediction, label)
        print(f'Error: {error}')

        # backpropagation
        delta = gradient(prediction, label)
        (weight, bias) = backward(hidden, delta, weight, bias)
        print(f"New weight: {weight}")
        print(f"New bias: {bias}")

        hidden_delta = gradient_backward(delta, weight)
        (hidden_weight, hidden_bias) = backward(feature, hidden_delta, hidden_weight, hidden_bias)
        print(f"New hidden weight: {hidden_weight}")
        print(f"New hidden bias: {hidden_bias}")
