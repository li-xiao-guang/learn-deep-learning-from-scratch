import numpy as np


# neuron definition
def forward(x, w, b):
    return x.dot(w.T) + b


# dataset
feature = np.array([28.1, 58.0])

# model
in_feature_size = feature.shape[-1]
out_feature_size = 1
weight = np.ones((out_feature_size, in_feature_size))
bias = np.zeros(out_feature_size)

# prediction
prediction = forward(feature, weight, bias)
print(f'Prediction: {prediction}')
