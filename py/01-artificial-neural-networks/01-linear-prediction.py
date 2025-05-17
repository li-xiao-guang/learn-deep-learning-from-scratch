import numpy as np


# neuron definition
def forward(x, w, b):
    return x.dot(w.T) + b


# dataset
feature = np.array([28.1, 58.0])

# model
feature_shape = feature.shape[-1], 1

weight = np.ones((feature_shape[1], feature_shape[0]))
bias = np.zeros(feature_shape[1])

# prediction
prediction = forward(feature, weight, bias)
print(f'Prediction: {prediction}')
