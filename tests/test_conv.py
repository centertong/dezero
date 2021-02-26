import unittest

import numpy as np
import matplotlib.pyplot as plt

import dezero
import dezero.functions as F


class ConvolutionTest(unittest.TestCase):
    def simple_conv(self):
        N, C, H, W = 1, 5, 15, 15
        OC, (KH, KW) = 8, (3, 3)

        x = Variable(np.random.randn(N, C, H, W))
        W = np.random.randn(OC, C, KH, KW)
        y = F.conv2d_simple(x, W, b=None, stride=1, pad=1)
        y.backward()

        print(y.shape)
        print(x.grad.shape)
