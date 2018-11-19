# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:23:25 2018

@author: Valeriy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ops_mlp1 import *
from torch.autograd import Variable
from genotypes_mlp import PRIMITIVES
from genotypes_mlp import Genotype


class MixedOp(nn.Module): ##initializing g'(x) = sum(softmax(gamma) g(x))

  def __init__(self, in_features, layers_size,out_size):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
        for sz in layers_size:
            
            op = OPS_NEW[primitive](in_features,int(sz),out_size)
            self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Network(nn.Module):
    def __init__(self,in_features,num_classes,layers_size,layers,out_shape,criterion): 
        super(Network, self).__init__()
        self._in_features = in_features
        self._num_classes = num_classes
        self._criterion = criterion
        self._layers = layers 
        self._layers_size = layers_size # layers size 2d array like ([in_1,out_1],[i2,o2])
                                        #where in_j and out_j are input and output sizes of layer j
        self._out_shape = out_shape
        self._initialize_alphas()
        self._ops = nn.ModuleList()
        op = MixedOp(self._in_features,self._layers_size,self._out_shape)
        self._ops.append(op)
        for i in range(self._layers-1):
            op = MixedOp(self._out_shape, self._layers_size,self._out_shape)
            self._ops.append(op)
        
        self._ops.append(Softmax(self._out_shape,self._num_classes,self._num_classes)) # softmax layer in the end to get normalized values
        
    def _initialize_alphas(self): #init gamma randomly
        k = self._layers
        num_ops = len(PRIMITIVES)*len(self._layers_size)
        self.alphas = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
          self.alphas
        ]
    def forward(self,x):  
        weights = F.softmax(self.alphas, dim=-1).cuda()
        x = x.view(-1,784)
        for i in range(self._layers):
            x = self._ops[i](x,weights[i,:])
        logits = self._ops[-1](x)
        return logits
    def new(self):
        model_new = Network(self._in_features, self._num_classes,self._layers_size,
                            self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new
    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target) 
    def arch_parameters(self):
        return self._arch_parameters
    def genotype(self):
        def _parse(weights):
            gene = []
            k = len(PRIMITIVES)*len(self._layers_size)
            for i in range(self._layers):
                w_max = 0
                j_max = 0
                for j in range(k):
                    if abs(weights[i,j]) > w_max:
                        w_max = abs(weights[i,j])
                        j_max = j
                gene.append([PRIMITIVES[(j+1)//len(PRIMITIVES) - 1],int(self._layers_size[(j+1) % len(PRIMITIVES) - 1])])
            return gene
        return _parse(self.alphas)
                        
                
            
            
    
    
    
    
    
