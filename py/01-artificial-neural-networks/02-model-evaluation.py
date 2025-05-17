import numpy as np


# neuron definition
def forward(x, w, b):
    return x.dot(w.T) + b


def mse_loss(p, y):
    return ((p - y) ** 2).mean()


# dataset
feature = np.array([28.1, 58.0])
label = np.array([165])

# model
weight = np.ones((label.shape[-1], feature.shape[-1]))
bias = np.zeros(label.shape[-1])

# prediction
prediction = forward(feature, weight, bias)
print(f'Prediction: {prediction}')

# evaluation
error = mse_loss(prediction, label)
print(f'Error: {error}')
