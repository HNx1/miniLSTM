import numpy as np


class GradFloat():
    def __init__(self, val, _children=[]):
        self.val = val
        self.grad = 0
        self._backward = lambda: None
        self._children = set(_children)

    def __repr__(self):
        return str(round(self.val, 2))

    def __add__(self, other):
        other = other if isinstance(other, GradFloat) else GradFloat(other)
        out = GradFloat(self.val+other.val, _children=(self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self+other

    def __mul__(self, other):
        other = other if isinstance(other, GradFloat) else GradFloat(other)
        out = GradFloat(self.val*other.val, _children=(self, other))

        def _backward():
            self.grad += out.grad*other.val
            other.grad += out.grad*self.val
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self*other

    def __sub__(self, other):
        return self.__add__(-other)

    def __pow__(self, other):
        out = GradFloat(self.val**other, _children=(self,))

        def _backward():
            self.grad += other*self.val**(other-1)*out.grad
        out._backward = _backward
        return out

    def tanh(self):
        out = GradFloat((np.exp(2*self.val)-1) /
                        (np.exp(2*self.val)+1), _children=(self,))

        def _backward():
            self.grad += (1-out.val**2)*out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        out = GradFloat(np.exp(self.val) /
                        (np.exp(self.val)+1), _children=(self,))

        def _backward():
            self.grad += (1-out.val)*out.val*out.grad
        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1.0
        topList = []
        seen = set()

        def topSort(v):
            if v not in seen:
                seen.add(v)
                for child in v._children:
                    topSort(child)
                topList.append(v)
        topSort(self)
        for item in reversed(topList):
            item._backward()


class LstmLayer():
    def __init__(self, inputSize, outputSize):
        self.wf = np.array([GradFloat(np.random.uniform(-1, 1))
                           for i in range(inputSize*outputSize)]).reshape(outputSize, inputSize)
        self.bf = [GradFloat(np.random.uniform(-1, 1))
                   for _ in range(outputSize)]
        self.wi = np.array([GradFloat(np.random.uniform(-1, 1))
                           for _ in range(inputSize*outputSize)]).reshape(outputSize, inputSize)
        self.bi = [GradFloat(np.random.uniform(-1, 1))
                   for _ in range(outputSize)]
        self.wC = np.array([GradFloat(np.random.uniform(-1, 1))
                           for _ in range(inputSize*outputSize)]).reshape(outputSize, inputSize)
        self.bC = [GradFloat(np.random.uniform(-1, 1))
                   for _ in range(outputSize)]
        self.wo = np.array([GradFloat(np.random.uniform(-1, 1))
                           for _ in range(inputSize*outputSize)]).reshape(outputSize, inputSize)
        self.bo = [GradFloat(np.random.uniform(-1, 1))
                   for _ in range(outputSize)]
        self.sz = outputSize

    def __call__(self, x, h, c):
        x = np.concatenate((h, x))
        f = np.array([y.sigmoid() for y in (np.matmul(
            self.wf, x)+self.bf).flatten()])
        i = np.array([y.sigmoid() for y in (np.matmul(
            self.wi, x)+self.bi).flatten()])
        C = np.array([y.tanh() for y in (np.matmul(
            self.wC, x)+self.bC).flatten()])
        o = np.array([y.sigmoid() for y in (np.matmul(
            self.wo, x)+self.bo).flatten()])
        c = f*c + i*C
        h = np.array([y.tanh() for y in c]) * o
        return h, c

    def parameters(self):
        params = self.bo + self.bC + self.bi + self.bf
        for mat in [self.wo, self.wC, self.wi, self.wf]:
            params.extend(mat.flatten())
        return params


class LSTM():
    def __init__(self, inputSize, outputSize, layers):
        n = LstmLayer(inputSize, outputSize)
        self.layers = [n for _ in range(layers)]
        self.sz = outputSize

    def __call__(self, xs):
        h = np.array([GradFloat(0)]*self.sz)
        c = np.array([GradFloat(0)]*self.sz)
        for x, layer in zip(xs, self.layers):
            h, c = layer(x, h, c)
        return np.sum(h)

    def parameters(self):
        return self.layers[0].parameters()

    def fullParameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def reset(self):
        params = self.parameters()
        for i, p in enumerate(self.fullParameters()):
            p.grad = 0
            p.val = params[i % len(params)].val

    def accumulate(self):
        params = self.parameters()
        for i, p in enumerate(self.fullParameters()):
            if i >= len(params):
                params[i % len(params)].grad += p.grad


def loss(model, data, truth):
    return sum(((y1-y2)**2) for y1, y2 in zip([model(x)for x in data], truth))


def train(model, data, truth, lr):
    model.reset()
    trainLoss = loss(model, data, truth)
    trainLoss.backward()
    model.accumulate()
    for p in model.parameters():
        p.val -= lr * p.grad


def test(model, data, truth):
    testLoss = loss(model, data, truth)
    print(f"Loss was {testLoss.val:.4f}")


def cycle(model, trainData, trainTruth, testData, testTruth, epochs=100, lr=1e-1):
    for _ in range(epochs):
        train(model, trainData, trainTruth, lr)
        test(model, testData, testTruth)
