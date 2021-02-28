import unittest

import numpy as np
import matplotlib.pyplot as plt

import dezero
import dezero.functions as F
import dezero.layers as L
#from dezero import Model

class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()
    
    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y


class RNNTest(unittest.TestCase):
    def simple_rnn(self):
        seq_data = [np.random.randn(1, 1) for _ in range(1000)]
        xs = seq_data[0:-1]
        ts = seq_data[1:]

        model = SimpleRNN(10, 1)

        loss, cnt = 0, 0
        for x, t in zip(xs, ts):
            y = model(x)
            loss += F.mean_squared_error(y, t)

            cnt += 1
            if cnt == 2:
                model.cleargrads()
                loss.backward()
                break

    
    def sin_train(self):
        max_epoch = 100
        hidden_size = 100
        bptt_length = 30
        train_set = dezero.datasets.SinCurve(train=True)
        seqlen = len(train_set)

        model = SimpleRNN(hidden_size, 1)
        optimizer = dezero.optimizers.Adam().setup(model)

        for epoch in range(max_epoch):
            model.reset_state()
            loss, count = 0, 0

            for x, t in train_set:
                x = x.reshape(1, 1)
                y = model(x)
                loss += F.mean_squared_error(y, t)
                count += 1

                if count % bptt_length == 0 or count == seqlen:
                    model.cleargrads()
                    loss.backward()
                    loss.unchain_backward()
                    optimizer.update()
                
            avg_loss = float(loss.data) / count
            print('| epoch %d | loss %f' % (epoch + 1, avg_loss))
