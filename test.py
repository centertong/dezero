import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
print('second')
y = f(x)
y.backward(create_graph=True)
print('second')

print(x.grad)
print('=================================')

print(x.grad.grad)
gx = x.grad
gx.backward()
print('second')

print(x.grad)
print('second')
