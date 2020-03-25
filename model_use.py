#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20-2-27 上午9:30
# @Author  : ivy_nie
# @File    : model_use.py
# @Software: PyCharm

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20-2-27 上午9:24
# @Author  : ivy_nie
# @File    : torch_test_save.py
# @Software: PyCharm

# Define model
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])

# torch.save(model.state_dict(), 'model2/save_model2.pkl')


    # print(name,param.requires_grad=True)


model = TheModelClass()
model_num=model.load_state_dict(torch.load('model2/save_model2.pkl'))
# #使用保存模型中的参数。 每次迭代打印该选项的话，会打印所有的name和param，但是这里的所有的param都是requires_grad=False,没有办法改变requires_grad的属性，所以改变requires_grad的属性只能通过上面的两种方式。
# for name, param in model.state_dict().items():
#     print(name)
#     print(param)
#model使用中的各个参数使用。迭代打印model.named_parameters()将会打印每一次迭代元素的名字和param
# for name, param in model.named_parameters():
#     print(name,param.requires_grad)
#     print(param)
##迭代打印model.parameters()将会打印每一次迭代元素的param而不会打印名字，这是他和named_parameters的区别，两者都可以用来改变requires_grad的属性
for  param in model.parameters():
    print(param.requires_grad)
# print(model_num)
# model_num=model.eval()
# print(model_num)
# print()
#
# # schemem1(recommended)
# print(model.state_dict()['1.weight'])

# scheme2
# params = list(model.named_parameters())  # get the index by debuging
#
# print(params)
# print(params[0][0])  # name
# print(params[2][1].data)  # data

# scheme3
# params = {}  # change the tpye of 'generator' into dict
# for name, param in model.named_parameters():
#     params[name] = param.detach().cpu().numpy()
# print(params['0.weight'])

# scheme4
for layer in model.modules():
    if (isinstance(layer, nn.Conv3d)):
        print(layer.weight)

# 打印每一层的参数名和参数值
# schemem1(recommended)
# for name, param in model.named_parameters():
#     print(name，param)
#
#     # scheme2
#     for name in model.state_dict():
#         print(name)
#         print(model.state_dict()[name])

