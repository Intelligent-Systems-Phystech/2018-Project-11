# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:19:04 2018

@author: Valeriy
"""

import torch
import torch.nn as nn


OPS_NEW = {
        'softmax' : lambda in_features,out_features : Softmax(in_features,out_features),
        'relu' : lambda in_features,out_features : ReLU(in_features,out_features),
        'tanh' : lambda in_features,out_features : Tanh(in_features,out_features)   
        }  ##list of available operations in each layer must be implemented as below


class Softmax(nn.Module): #softmax(w.transpose*x)
    def __init__(self,in_features,out_features):
        super(Softmax,self).__init__()
        self.op = nn.Sequential(
            nn.Linear(in_features,out_features),
            nn.Softmax()
            )
    def forward(self,x):
        return self.op(x)

class ReLU(nn.Module):
    def __init__(self,in_features,out_features):
        super(ReLU,self).__init__()
        self.op = nn.Sequential(
            nn.Linear(in_features,out_features),
            nn.ReLU()
            )
    def forward(self,x):
        return self.op(x)   
class Tanh(nn.Module):
    def __init__(self,in_features,out_features):
        super(Tanh,self).__init__()
        self.op = nn.Sequential(
            nn.Linear(in_features,out_features),
            nn.Tanh()
            )
    def forward(self,x):
        return self.op(x)







    