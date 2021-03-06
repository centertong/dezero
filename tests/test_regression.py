import unittest

import numpy as np
import matplotlib.pyplot as plt

from dezero import Variable
import dezero.functions as F
import dezero.layers as L
from dezero.utils import get_dot_graph
from dezero import optimizers
from dezero import models

class RegressionTest(unittest.TestCase):
    def test_linear_regressoion(self):
        np.random.seed(0)
        x = np.random.rand(100, 1)
        y = 5 + 2 * x + np.random.rand(100,1)
        x, y = Variable(x), Variable(y)

        W = Variable(np.zeros((1,1)))
        b = Variable(np.zeros(1))

        def predict(x):
            y = F.matmul(x, W) + b
            return y
        
        def mean_squared_error(x0, x1):
            diff = x0 - x1
            return F.sum(diff ** 2) / len(diff)

        lr = 0.1
        iters = 100

        for i in range(iters):
            y_pred = predict(x)
            loss = F.mean_squared_error(y, y_pred)

            W.cleargrad()
            b.cleargrad()
            loss.backward()

            W.data -= lr * W.grad.data
            b.data -= lr * b.grad.data
        
        print(W, b, loss)
    
    def test_neural_regression(self):
        np.random.seed(0)
        x = np.random.rand(100,1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100,1)

        I, H, O = 1, 10, 1
        W1 = Variable(0.01 * np.random.randn(I, H))
        b1 = Variable(np.zeros(H))
        W2 = Variable(0.01 + np.random.randn(H, O))
        b2 = Variable(np.zeros(O))

        def predict(x):
            y = F.linear(x, W1, b1)
            y = F.sigmoid(y)
            y = F.linear(y, W2, b2)
            return y
        
        lr = 0.2
        iters = 10000

        for i in range(iters):
            y_pred = predict(x)
            loss = F.mean_squared_error(y, y_pred)

            W1.cleargrad()
            b1.cleargrad()
            W2.cleargrad()
            b2.cleargrad()
            loss.backward()

            W1.data -= lr * W1.grad.data
            b1.data -= lr * b1.grad.data
            W2.data -= lr * W2.grad.data
            b2.data -= lr * b2.grad.data
            
    def test_neural_regression_layer(self):
        np.random.seed(0)
        x = np.random.rand(100,1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100,1)

        model = L.Layer()
        model.l1 = L.Linear(10)
        model.l2 = L.Linear(1)

        def predict(model, x):
            y = model.l1(x)
            y = F.sigmoid(y)
            y = model.l2(y)
            return y
        
        lr = 0.2
        iters = 10000

        for i in range(iters):
            y_pred = predict(model, x)
            loss = F.mean_squared_error(y, y_pred)

            model.cleargrads()
            loss.backward()

            for p in model.params():
                p.data -= lr * p.grad.data
            
            #if i % 1000 == 0:
            #    print(loss)            
    
    def test_sgd(self):
        np.random.seed(0)
        x = np.random.rand(100,1)
        y = np.sin(2 * np.pi * x) + np.random.rand(100,1)

        lr = 0.2
        max_iter = 10000
        hidden_size = 10

        model = models.MLP((hidden_size, 1))
        optimizer = optimizers.SGD(lr)
        optimizer.setup(model)

        for i in range(max_iter):
            y_pred = model(x)
            loss = F.mean_squared_error(y, y_pred)

            model.cleargrads()
            loss.backward()

            optimizer.update()
            if i % 1000 == 0:
                print(loss)
                