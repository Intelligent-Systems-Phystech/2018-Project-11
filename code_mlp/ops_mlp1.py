# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:19:04 2018

@author: Valeriy
"""

import torch
import torch.nn as nn


OPS_NEW = {
        'softmax' : lambda in_features,out_features, out_size : Softmax(in_features,out_features,out_size),
        'relu' : lambda in_features,out_features, out_size : ReLU(in_features,out_features, out_size),
        'tanh' : lambda in_features,out_features, out_size : Tanh(in_features,out_features, out_size)   
        }  ##list of available operations in each layer must be implemented as below


class Softmax(nn.Module): #softmax(w.transpose*x)
    def __init__(self,in_features,out_features,out_size):
        super(Softmax,self).__init__()
        self.out_size = out_size
        self.out_features = out_features
        self.op = nn.Sequential(
            nn.Linear(in_features,out_features),
            nn.Softmax()
            )
    def forward(self,x):
        print(self.out_size,"  ",self.out_features )
        if self.out_size != self.out_features:
            return torch.cat((self.op(x),torch.zeros(x.size()[0],self.out_size-self.out_features).cuda()), dim = 1).cuda()
        else:
            return self.op(x)

class ReLU(nn.Module):
    def __init__(self,in_features,out_features,out_size):
        super(ReLU,self).__init__()
        self.out_size = out_size
        self.out_features = out_features        
        self.op = nn.Sequential(
            nn.Linear(in_features,out_features),
            nn.ReLU()
            )
    def forward(self,x):
        if self.out_size != self.out_features:
            return torch.cat((self.op(x),torch.zeros(x.size()[0],self.out_size-self.out_features).cuda()), dim = 1).cuda()
        else:
            return self.op(x)
class Tanh(nn.Module):
    def __init__(self,in_features,out_features,out_size):
        super(Tanh,self).__init__()
        self.out_size = out_size
        self.out_features = out_features
        self.op = nn.Sequential(
            nn.Linear(in_features,out_features),
            nn.Tanh()
            )
    def forward(self,x):
        if self.out_size != self.out_features:
            return torch.cat((self.op(x),torch.zeros(x.size()[0],self.out_size-self.out_features).cuda()), dim = 1).cuda()
        else:
            return self.op(x)