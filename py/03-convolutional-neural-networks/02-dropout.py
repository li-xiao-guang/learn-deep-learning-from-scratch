from abc import ABC, abstractmethod

import numpy as np


def merge_grad(old, new):
    return new if old is None else (old + new)


class Tensor:

    def __init__(self, data, requires_grad=True):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.backward_fn = lambda: None
        self.parents = set()

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

    def __init__(self):
        self.training = True

    def __call__(self, x: Tensor):
        return self.forward(x)

    @abstractmethod
    def forward(self, x: Tensor):
        pass

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


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


class Flatten(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.array(x.data.reshape(x.data.shape[0], -1)))

        def backward_fn():
            if x.requires_grad:
                grad = p.grad.reshape(x.data.shape)
                x.grad = merge_grad(x.grad, grad)

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


class Dropout(Layer):

    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x: Tensor):
        if not self.training:
            return x

        mask = np.random.random(x.data.shape) > self.dropout_rate
        p = Tensor(x.data * mask)

        def backward_fn():
            if x.requires_grad:
                grad = p.grad * mask
                x.grad = merge_grad(x.grad, grad)

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


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


class ReLU(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.maximum(0, x.data))

        def backward_fn():
            if x.requires_grad:
                grad = (p.data > 0) * p.grad
                x.grad = merge_grad(x.grad, grad)

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


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

    def __init__(self, test=False, size=2000, batch_size=1):
        self.batch_size = batch_size

        with (np.load('mnist.npz', allow_pickle=True) as f):
            if test:
                x, y = f['x_test'][:size], f['y_test'][:size]
            else:
                x, y = f['x_train'][:size], f['y_train'][:size]
        self.features, self.labels = self.normalize(x, y)

    @staticmethod
    def normalize(x, y):
        inputs = x / 255
        targets = np.zeros((len(y), 10))
        targets[range(len(y)), y] = 1
        return inputs, targets

    def count(self):
        return len(self.features)

    def feature(self, index):
        return Tensor(self.features[index: index + self.batch_size])

    def feature_row(self):
        return self.feature(0).data.shape[-2]

    def feature_column(self):
        return self.feature(0).data.shape[-1]

    def feature_size(self):
        return self.feature_row() * self.feature_column()

    def label(self, index):
        return Tensor(self.labels[index: index + self.batch_size])

    def label_size(self):
        return self.label(0).data.shape[-1]


np.random.seed(99)

LEARNING_RATE = 0.1
EPOCHES = 10
BATCHES = 2

# training
dataset = Dataset(batch_size=BATCHES)

hidden = Linear(dataset.feature_size(), 64)
output = Linear(64, dataset.label_size())
model = Sequential([Flatten(), Dropout(), hidden, output])

loss = MSELoss()
optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHES):
    print(f"Epoch: {epoch}")

    for i in range(0, dataset.count(), BATCHES):
        feature = dataset.feature(i)
        label = dataset.label(i)

        prediction = model(feature)
        error = loss(prediction, label)

        optimizer.zero_grad()
        error.backward()
        optimizer.step()

    print(f'Prediction: {prediction.data}')
    print(f'Error: {error.data.item()}')
    print(f"New weight: {output.weight.data}")
    print(f"New bias: {output.bias.data}")
    print(f"New hidden weight: {hidden.weight.data}")
    print(f"New hidden bias: {hidden.bias.data}")

# evaluation
dataset = Dataset(True, 1000, batch_size=BATCHES)

model.eval()

result = 0
for i in range(0, dataset.count(), BATCHES):
    feature = dataset.feature(i)
    label = dataset.label(i)

    prediction = model(feature)
    for j in range(BATCHES):
        if prediction.data[j].argmax() == label.data[j].argmax():
            result += 1

print(f'Result: {result} of {dataset.count()}')
