import unittest

import numpy as np
from dezero import Variable
from dezero.utils import get_dot_graph

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
        if isinstance(x.grad, Variable):
            self.assertEqual(x.grad.data, expected)
            self.assertEqual(y.grad.data, expected)
        else:
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
        if isinstance(x.grad, Variable):
            self.assertEqual(x.grad.data, expected)
            self.assertEqual(y.grad.data, expected)
        else:
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
        if isinstance(x.grad, Variable):
            self.assertEqual(x.grad.data, expected0)
            self.assertEqual(y.grad.data, expected1)
        else:
            self.assertEqual(x.grad, expected0)
            self.assertEqual(y.grad, expected1)
        
    
    def test_rosenbrock(self):
        def rosenbrock(x0, x1, a = 1, b=100):
            y  = b * (x1 - x0 ** 2) ** 2 + (a - x0) ** 2
            return y
        
        x0 = Variable(np.array(0.0))
        x1 = Variable(np.array(2.0))

        lr = 0.001
        iters = 50000
        for i in range(iters):
            y = rosenbrock(x0, x1)
            
            x0.cleargrad()
            x1.cleargrad()
            y.backward()
            
            x0.data -= lr * x0.grad
            x1.data -= lr * x1.grad
        
        flg1 = np.allclose(x0.data, 1.0)
        flg2 = np.allclose(x1.data, 1.0)
        self.assertTrue(flg1 and flg2)

    def test_newton_gd(self):
        def f(x):
            y = x ** 4 - 2 * x ** 2
            return y

        def gx2(x):
            return 12 * x ** 2 - 4
        
        x = Variable(np.array(2.0))
        iters = 10

        for i in range(iters):
            print(i, x)
            y = f(x)
            x.cleargrad()
            y.backward()

            x.data -= x.grad / gx2(x.data)

