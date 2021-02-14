import unittest

import numpy as np
import matplotlib.pyplot as plt

from dezero import Variable
import dezero.functions as F
from dezero.utils import get_dot_graph


class SecondOrder(unittest.TestCase):
    def test_newton(self):
        def f(x):
            y = x ** 4 - 2 * x ** 2
            return y

        x = Variable(np.array(2.0))
        iters = 10

        for i in range(iters):
            y = f(x)
            x.cleargrad()
            y.backward(create_graph=True)

            gx = x.grad
            x.cleargrad()
            gx.backward()
            gx2 = x.grad

            x.data -= gx.data / gx2.data

    def test_sin(self):
        x = Variable(np.linspace(-7, 7, 200))
        y = F.sin(x)
        y.backward(create_graph=True)

        logs = [y.data]

        for i in range(3):
            logs.append(x.grad.data)
            gx = x.grad
            x.cleargrad()
            gx.backward(create_graph=True)
        
        labels = ["y=sin(x)", "y'", "y''", "y'''"]
        for i, v in enumerate(logs):
            plt.plot(x.data, logs[i], label= labels[i])
        plt.legend(loc='lower right')
        plt.savefig('test.png')
    
    def test_tanh(self):
        x = Variable(np.array(1.0))
        y = F.tanh(x)
        x.name = 'x'
        y.name = 'y'
        y.backward(create_graph=True)

        iters = 0

        for i in range(iters):
            gx = x.grad
            x.cleargrad()
            gx.backward(create_graph=True)
        
        gx = x.grad
        gx.name = 'gx' + str(iters+1)
        txt = get_dot_graph(gx)
        with open('test.dot', 'w') as f:
            f.write(txt)

    def test_double_backprop(self):
        def y_(x):
            y = x ** 2
            return y
        
        def z_(gx, y):
            z = gx ** 3 + y
            return z
        
        x = Variable(np.array(2.0))
        y = y_(x)
        y.backward(create_graph=True)
        
        gx = x.grad
        x.cleargrad()
        z = z_(gx, y)
        z.backward()
        print(x.grad)
