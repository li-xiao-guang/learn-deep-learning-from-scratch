from abc import ABC, abstractmethod

import numpy as np

np.random.seed(99)

LEARNING_RATE = 0.00001
EPOCHES = 1000
BATCH_SIZE = 2


def merge_grad(old, new):
    return new if old is None else (old + new)


class Tensor:

    def __init__(self, data, requires_grad=True):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.backward_fn = lambda: None
        self.parents = set()

    def shape(self, axis):
        return self.data.shape[axis] if axis else self.data.shape

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad

        if self.backward_fn:
            self.backward_fn()

        for p in self.parents:
            if p.requires_grad:
                p.backward(p.grad)


class Layer(ABC):

    def __call__(self, x: Tensor):
        return self.forward(x)

    @abstractmethod
    def forward(self, x: Tensor):
        pass

    def parameters(self):
        return []


class Linear(Layer):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(np.random.rand(out_features, in_features) / in_features)
        self.bias = Tensor(np.zeros(out_features))

    def forward(self, x: Tensor):
        p = Tensor(x.data.dot(self.weight.data.T) + self.bias.data)

        def backward_fn():
            if self.weight.requires_grad:
                grad = p.grad.T.dot(x.data)
                self.weight.grad = merge_grad(self.weight.grad, grad)
            if self.bias.requires_grad:
                grad = np.sum(p.grad, axis=0)
                self.bias.grad = merge_grad(self.bias.grad, grad)
            if x.requires_grad:
                grad = p.grad.dot(self.weight.data)
                x.grad = merge_grad(x.grad, grad)

        p.backward_fn = backward_fn
        p.parents = {self.weight, self.bias, x}
        return p

    def parameters(self):
        return [self.weight, self.bias]


class Sequential(Layer):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]


class MSELoss:

    def __call__(self, p: Tensor, y: Tensor):
        mse = Tensor(((p.data - y.data) ** 2).mean())

        def backward_fn():
            if p.requires_grad:
                p.grad = (p.data - y.data) * 2 / np.prod(y.data.shape)

        mse.backward_fn = backward_fn
        mse.parents = {p}
        return mse


class SGD:

    def __init__(self, params, lr):
        self.parameters = params
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            if p is not None and p.grad is not None:
                p.grad = None

    def step(self):
        for p in self.parameters:
            if p is not None and p.grad is not None:
                p.data -= p.grad.reshape(p.data.shape) * self.lr


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
        self.shape = len(self.features[0]), len(self.labels[0])

    def size(self):
        return len(self.features)

    def feature(self, index):
        return Tensor(self.features[index: index + BATCH_SIZE])

    def label(self, index):
        return Tensor(self.labels[index: index + BATCH_SIZE])


dataset = Dataset()

hidden = Linear(dataset.shape[0], 4)
output = Linear(4, dataset.shape[1])
model = Sequential([hidden, output])

loss = MSELoss()
optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHES):
    print(f"Epoch: {epoch}")

    for i in range(0, dataset.size(), BATCH_SIZE):
        feature = dataset.feature(i)
        label = dataset.label(i)

        prediction = model(feature)
        print(f'Prediction: {prediction.data}')

        error = loss(prediction, label)
        print(f'Error: {error.data.item()}')

        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        print(f"New weight: {output.weight.data}")
        print(f"New bias: {output.bias.data}")
        print(f"New hidden weight: {hidden.weight.data}")
        print(f"New hidden bias: {hidden.bias.data}")
