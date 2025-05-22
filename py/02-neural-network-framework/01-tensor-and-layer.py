import numpy as np


class Tensor:

    def __init__(self, data):
        self.data = np.array(data)


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
        self.feature = Tensor([28.1, 58.0])
        self.label = Tensor([165])

    def feature_size(self):
        return self.feature.data.shape[-1]

    def label_size(self):
        return self.label.data.shape[-1]


dataset = Dataset()

model = Linear(dataset.feature_size(), dataset.label_size())

loss = MSELoss()

prediction = model(dataset.feature)
print(f'Prediction: {prediction.data}')

error = loss(prediction, dataset.label)
print(f'Error: {error.data.item()}')
