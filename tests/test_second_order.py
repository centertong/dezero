import unittest

import numpy as np
from dezero import Variable

class SecondOrder(unittest.TestCase):
    def test_second_order(self):
        def f(x):
            y = x ** 4 - 2 * x ** 2
            return y

        x = Variable(np.array(2.0))
        print('second')
        y = f(x)
        y.backward(create_graph=True)
        print(x.grad)

        gx = x.grad
        gx.backward()
        print(x.grad)
        print('second')

        
