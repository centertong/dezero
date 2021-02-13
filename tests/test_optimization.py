import unittest

import numpy as np
from dezero import Variable


class Optimization(unittest.TestCase):
    def test_sphere(self):
        def sphere(x,y):
            z = x ** 2 + y ** 2
            return z
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = sphere(x,y)
        z.backward()
        expected = np.array(2.0)
        self.assertEqual(x.grad, expected)
        self.assertEqual(y.grad, expected)
    
    def test_matyas(self):
        def matyas(x,y):
            z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
            return z
        
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = matyas(x,y)
        z.backward()
        expected = np.array(0.040000000000000036)
        self.assertEqual(x.grad, expected)
        self.assertEqual(y.grad, expected)
    
    def test_goldstein(self):
        def goldstein(x,y):
            z = (1 + (x + y + 1) ** 2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
                (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
            return z
        
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = goldstein(x,y)
        z.backward()
        expected0 = np.array(-5376.0)
        expected1 = np.array(8064.0)
        self.assertEqual(x.grad, expected0)
        self.assertEqual(y.grad, expected1)