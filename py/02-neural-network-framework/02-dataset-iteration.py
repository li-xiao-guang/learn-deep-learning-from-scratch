import numpy as np


class Tensor:

    def __init__(self, data):
        self.data = np.array(data)

    def shape(self, axis):
        return self.data.shape if axis is None else self.data.shape[axis]


class Linear:

    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Tensor(np.ones((out_features, in_features)))
        self.bias = Tensor(np.zeros(out_features))

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        return Tensor(x.data.dot(self.weight.data.T) + self.bias.data)


class MSELoss:

    def __call__(self, p: Tensor, y: Tensor):
        return Tensor(((p.data - y.data) ** 2).mean())


class Dataset:

    def __init__(self):
        self.features = [[28.1, 58.0],
                         [22.5, 72.0],
                         [31.4, 45.0],
                         [19.8, 85.0],
                         [27.6, 63]]
        self.labels = [[165],
                       [95],
                       [210],
                       [70],
                       [155]]

    def size(self):
        return len(self.features)

    def feature(self, index):
        return Tensor(self.features[index])

    def label(self, index):
        return Tensor(self.labels[index])

    def feature_size(self):
        return self.feature(0).shape(-1)

    def label_size(self):
        return self.label(0).shape(-1)

dataset = Dataset()

model = Linear(dataset.feature_size(), dataset.label_size())

loss = MSELoss()

for i in range(dataset.size()):
    feature = dataset.feature(i)
    label = dataset.label(i)

    prediction = model(feature)
    print(f'Prediction: {prediction.data}')

    error = loss(prediction, label)
    print(f'Error: {error.data.item()}')
