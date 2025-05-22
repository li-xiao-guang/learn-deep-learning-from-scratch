import numpy as np


# neuron definition
def forward(x, w, b):
    return x.dot(w.T) + b


# loss function
def mse_loss(p, y):
    return ((p - y) ** 2).mean()


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
