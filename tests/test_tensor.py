import unittest

import numpy as np
import matplotlib.pyplot as plt

from dezero import Variable
import dezero.functions as F
from dezero.utils import get_dot_graph


class TensorTest(unittest.TestCase):
    def test_sum(self):
        x = Variable(np.array([1,2,3,4,5,6]))
        y = F.sum(x)
        y.backward()

        expected = np.array([1,1,1,1,1,1])
        flg = np.alltrue(x.grad.data == expected)
        self.assertTrue(flg)
        
        x = Variable(np.array([[1,2,3],[4,5,6]]))
        y = F.sum(x)
        y.backward()

        expected = np.array([[1,1,1],[1,1,1]])
        flg = np.alltrue(x.grad.data == expected)
        self.assertTrue(flg)
        
    def test_add(self):
        x0 = Variable(np.array([1,2,3]))
        x1 = Variable(np.array([10]))
        y = x0 + x1
        expectedy = np.array([11,12,13])
        flg = np.alltrue(y.data == expectedy)
        self.assertTrue(flg)
        
        y.backward()
        expectedgx = np.array([3])
        flg = np.alltrue(x1.grad.data == expectedgx)
        self.assertTrue(flg)
        
    def test_matmul(self):
        x = Variable(np.random.randn(2,3))
        W = Variable(np.random.randn(3,4))
        y = F.matmul(x, W)
        y.backward()

        print(x.grad.shape)
        print(W.grad.shape)