#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20-2-27 下午2:05
# @Author  : ivy_nie
# @File    : classify.py
# @Software: PyCharm
import sklearn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim
from sklearn import datasets

#
iris = datasets.load_iris()

x = iris['data']
y = iris['target']
x = torch.FloatTensor(x)
y = torch.LongTensor(y)
x = Variable(x)

y = Variable(y)


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_out):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.out = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = F.sigmoid(self.hidden(x))
        x = self.out(x)
        out = F.log_softmax(x, dim=1)
        return out


net = Net(n_feature=4, n_hidden=5, n_out=4)

# optimizer = torch.optim.SGD(net.parameters(), lr=0.03)
#
# epochs = 10000
#
# px = [];
# py = []
# for i in range(epochs):
#     predict = net(x)
#     loss = F.nll_loss(predict, y)  # 输出层 用了log_softmax 则需要用这个误差函数
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(i, "loss:", loss.item())
#     px.append(i)
#     py.append(loss.item())
    # if(i%10 == 0):
    #      plt.cla
    #      plt.title(u"训练过程的loss曲线")
    #      plt.xlabel(u"迭代次数")
    #      plt.ylabel(u"迭代损失")
    #      plt.plot(px,py,"r-",lw=1)
    #      plt.text(0,0,"Loss = %.4f" % loss.data[0],fontdict={"size":20,'color':'red'})
    # plt.pause(0.1)
# torch.save(net, "model3/my_model.pkl")

# iris_model = torch.load("iris_model.pkl")
# print(iris_model)
net = torch.load("model3/my_model.pkl")

x1 = torch.FloatTensor([5.1000, 3.5000, 1.4000, 0.2000])
x1 = Variable(x1)
x2 = Variable(torch.FloatTensor([4.9000, 3.0000, 1.4000, 0.2000]))
# print(x1)
print(x1.unsqueeze(0))
print(net(x1.unsqueeze(0)))  # 单独一个样板 需要 unsqueeze(0)
print(net(x2.unsqueeze(0)))
x = iris['data']
# print(x)
x = Variable(torch.FloatTensor(x))
print(iris['data'])
all_predict = net(x).data.numpy()

'''
    argmax(data,axis = 1)  axis = 1表示 按照行求最大值的索引
'''
print((np.argmax(all_predict,axis=1) == iris['target']).sum()/len(y))
