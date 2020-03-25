#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20-2-27 上午4:16
# @Author  : ivy_nie
# @File    : testload.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from TextCNN_model import TextCNN_model

# model=TextCNN_model()
#
# print("model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, '\t', model.state_dict()[param_tensor].size())

# print("\noptimizer's state_dict")
# for var_name in optimizer.state_dict():
#     print(var_name, '\t', optimizer.state_dict()[var_name])
#
# print("\nprint particular param")
# print('\n', model.conv1.weight.size())
# print('\n', model.conv1.weight)
#
# print("------------------------------------")
# torch.save(model.state_dict(), './model_state_dict.pt')
# model_2 = TheModelClass()
# model_2.load_state_dict(torch.load('./model_state_dict'))
# model.eval()
# print('\n',model_2.conv1.weight)
# print((model_2.conv1.weight == model.conv1.weight).size())
## 仅仅加载某一层的参数
conv1_weight_state = torch.load('model/CNN_model.pth')
print(conv1_weight_state)
# print(conv1_weight_state == model.conv1.weight)

# model_2 = TheModelClass()
# model_2.load_state_dict(torch.load('./model_state_dict.pt'))
# model_2.conv1.requires_grad = False
# print(model_2.conv1.requires_grad)
# print(model_2.conv1.bias.requires_grad)
