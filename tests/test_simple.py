import unittest

import numpy as np
from dezero import Variable
from dezero.core import mul, add, square, exp, neg, sin
import math

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
        

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        if isinstance(x.grad, Variable):
            self.assertEqual(x.grad.data, expected)
        else:
            self.assertEqual(x.grad, expected)
        

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        if isinstance(x.grad, Variable):
            flg = np.allclose(x.grad.data, num_grad)
        else:
            flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

    def test_taylor(self):
        def my_sin(x, threshold=0.00001):
            y = 0
            for i in range(100000):
                c = (-1) ** i / math.factorial(2 * i + 1)
                t = c * x ** (2 * i + 1)
                y = y + t
                if abs(t.data) < threshold:
                    break
            return y
        
        x = Variable(np.array(np.pi / 4))
        y = my_sin(x)
        y.backward()
        if isinstance(x.grad, Variable):
            flg = np.allclose(y.data, x.grad.data)
        else:
            flg = np.allclose(y.data, x.grad)
        self.assertTrue(flg)


#unittest.main()
