import numpy as np


# neuron definition
def forward(x, w, b):
    return x.dot(w.T) + b


# dataset
feature = np.array([28.1, 58.0])

# model
weight = np.ones((1, feature.shape[-1]))
bias = np.zeros(1)

# prediction
prediction = forward(feature, weight, bias)
print(f'Prediction: {prediction}')
