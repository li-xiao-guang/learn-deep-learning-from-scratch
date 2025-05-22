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

    def backward(self):
        if self.backward_fn:
            self.backward_fn()

        for p in self.parents:
            if p.requires_grad:
                p.backward()


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
                x.grad = p.grad.dot(self.weight.data)

        p.backward_fn = backward_fn
        p.parents = {self.weight, self.bias, x}
        return p

    def parameters(self):
        return [self.weight, self.bias]


class Convolution2D(Layer):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        in_size = kernel_size ** 2 * in_channels
        self.weight = Tensor(np.random.rand(out_channels, in_size) / in_size)
        self.bias = Tensor(np.zeros(out_channels))

    def forward(self, x: Tensor):
        batch, channel, row, column = x.data.shape
        row = row - self.kernel_size + 1
        column = column - self.kernel_size + 1

        patches = []
        for b in range(batch):
            for c in range(channel):
                for r in range(row):
                    for l in range(column):
                        patch = x.data[b,
                                c:c + self.in_channels,
                                r:r + self.kernel_size,
                                l:l + self.kernel_size]
                        patches.append(patch)
        patches = np.array(patches).reshape(batch, channel, row, column, -1)

        p = Tensor(patches.dot(self.weight.data.T) + self.bias.data)

        def backward_fn():
            if self.weight.requires_grad:
                grad = p.grad.reshape(-1, self.out_channels).T.dot((patches.reshape(-1, self.kernel_size ** 2)))
                self.weight.grad = merge_grad(self.weight.grad, grad)
            if self.bias.requires_grad:
                grad = np.sum(p.grad.reshape(-1, self.out_channels), axis=0)
                self.bias.grad = merge_grad(self.bias.grad, grad)

        p.backward_fn = backward_fn
        p.parents = {self.weight, self.bias}
        return p

    def parameters(self):
        return [self.weight]


class Pool2D(Layer):

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: Tensor):
        batch, channel, row, column, patch = x.data.shape
        pooled_row = row // self.kernel_size
        pooled_column = column // self.kernel_size

        masks = np.zeros_like(x.data, dtype=bool)
        pools = np.zeros((batch, channel, pooled_row, pooled_column, patch))
        for r in range(pooled_row):
            for l in range(pooled_column):
                row_slice = slice(r * self.kernel_size, (r + 1) * self.kernel_size)
                column_slice = slice(l * self.kernel_size, (l + 1) * self.kernel_size)
                region = x.data[:, :, row_slice, column_slice, :]
                max_region = region.max(axis=(2, 3), keepdims=True)
                pools[:, :, r, l, :] = max_region.squeeze(axis=(2, 3))
                mask = region == max_region
                masks[:, :, row_slice, column_slice, :] += mask

        p = Tensor(pools)

        def backward_fn():
            if x.requires_grad:
                x.grad = np.zeros_like(x.data)
                x.grad[masks] = p.grad.repeat(self.kernel_size, axis=2).repeat(self.kernel_size, axis=3)[masks]

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


class Flatten(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.array(x.data.reshape(x.data.shape[0], -1)))

        def backward_fn():
            if x.requires_grad:
                x.grad = p.grad.reshape(x.data.shape)

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
                x.grad = p.grad * mask

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
                x.grad = (p.data > 0) * p.grad

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


class Tanh(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.tanh(x.data))

        def backward_fn():
            if x.requires_grad:
                x.grad = p.grad * (1 - p.data ** 2)

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


class Sigmoid(Layer):

    def __init__(self, clip_range=(-100, 100)):
        super().__init__()
        self.clip_range = clip_range

    def forward(self, x: Tensor):
        z = np.clip(x.data, self.clip_range[0], self.clip_range[1])
        p = Tensor(1 / (1 + np.exp(-z)))

        def backward_fn():
            if x.requires_grad:
                x.grad = p.grad * p.data * (1 - p.data)

        p.backward_fn = backward_fn
        p.parents = {x}
        return p


class Softmax(Layer):

    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor):
        exp = np.exp(x.data - np.max(x.data, axis=self.axis, keepdims=True))
        p = Tensor(exp / np.sum(exp, axis=self.axis, keepdims=True))

        def backward_fn():
            if x.requires_grad:
                x.grad = np.zeros_like(x.data)
                for idx in range(x.data.shape[0]):
                    itm = p.data[idx].reshape(-1, 1)
                    x.grad[idx] = (np.diagflat(itm) - itm.dot(itm.T)).dot(p.grad[idx])

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
        inputs = np.expand_dims(x / 255, axis=1)
        targets = np.zeros((len(y), 10))
        targets[range(len(y)), y] = 1
        return inputs, targets

    def count(self):
        return len(self.features)

    def feature(self, index):
        return Tensor(self.features[index: index + self.batch_size])

    def feature_channel(self):
        return self.feature(0).data.shape[-3]

    def feature_row(self):
        return self.feature(0).data.shape[-2]

    def feature_column(self):
        return self.feature(0).data.shape[-1]

    def feature_size(self):
        return self.feature_channel() * self.feature_row() * self.feature_column()

    def label(self, index):
        return Tensor(self.labels[index: index + self.batch_size])

    def label_size(self):
        return self.label(0).data.shape[-1]


np.random.seed(99)

LEARNING_RATE = 0.1
EPOCHES = 10
BATCHES, CHANNELS, KERNELS, POOLS = (2, 16, 3, 2)

# training
dataset = Dataset(batch_size=BATCHES)

kernel = Convolution2D(dataset.feature_channel(), CHANNELS, KERNELS)
pool = Pool2D(POOLS)
convolved_row = dataset.feature_row() - KERNELS + 1
convolved_column = dataset.feature_column() - KERNELS + 1
hidden = Linear((convolved_row // POOLS) * (convolved_column // POOLS) * CHANNELS, 64)
output = Linear(64, dataset.label_size())
model = Sequential([kernel, pool, Flatten(), Dropout(), hidden, Tanh(), output, Softmax()])

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
