#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 20-2-27 上午9:02
# @Author  : ivy_nie
# @File    : test_net_.py
# @Software: PyCharm

# 定义模型
import data_preprocess
import torch.nn as nn
import torch.nn.functional as F
import torch

#定义模型
class LSTM_model(nn.Module):
    def __init__(self,len_dic,emb_dim):
        super(LSTM_model,self).__init__()
        self.embed=nn.Embedding(len_dic,emb_dim)  #b,64,128  -> 64,b,128
        self.lstm1=nn.LSTM(input_size=emb_dim,hidden_size=256,dropout=0.2)#64,b,256
        self.lstm2=nn.LSTM(input_size=256,hidden_size=256,dropout=0.2)#64,b,256 -> b,256
        self.classify=nn.Linear(256,3)#b,3

    def forward(self, x):
        x=self.embed(x)
        # print(x.size())
        x=x.permute(1,0,2)
        out,_=self.lstm1(x)
        out,_=self.lstm2(out)
        out=out[-1,:,:]
        # print(out.size())
        out=out.view(-1,256)
        out=self.classify(out)
        # print(out.size())
        return out
# 获取字典
word_to_inx, inx_to_word = data_preprocess.get_dic()
len_dic = len(word_to_inx)
MAXLEN = 64
input_dim = MAXLEN
emb_dim = 128
num_epoches = 20
batch_size = 16

model=LSTM_model(len_dic,emb_dim)
model.load_state_dict(torch.load('model/LSTM_model.pth'))

# model.load_state_dict(torch.load('model/CNN_model.pth'))
print(model.eval())
for name, param in model.state_dict().items():
    print(name)
    print(param)
