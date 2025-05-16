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
in_feature_size = feature.shape[-1]
out_feature_size = label.shape[-1]
weight = np.ones((out_feature_size, in_feature_size))
bias = np.zeros(out_feature_size)

# prediction
prediction = forward(feature, weight, bias)
print(f'Prediction: {prediction}')

# evaluation
error = mse_loss(prediction, label)
print(f'Error: {error}')
