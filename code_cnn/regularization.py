from math import pow, factorial, log
from math import gamma as gamma_func

import torch 
import torch.nn.functional as F

import warnings

import torch
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import LogSoftmax


from torch.nn.functional import _Reduction

from torch.nn.modules.loss import _WeightedLoss


class CrossEntropyLoss_GumbelSoftmaxRegularization(_WeightedLoss):
    def regularization_gumbel_softmax(self, gamma, temperature, alpha):
        K = len(gamma)

        tmp = 1.0

        for i in range(K):
            denom = alpha
            acc = 0.0
            for j in range(K):
                acc += pow(gamma[j], -temperature)
            denom *= acc
            tmp *= (pow(gamma[i], -temperature - 1) / denom)

        tmp *= factorial(K - 1) * pow(temperature, K - 1) * pow(alpha, K)
        return log(tmp, 2)

    def __init__(self, tau, alpha, reg_coef, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='elementwise_mean'):
        super(CrossEntropyLoss_GumbelSoftmaxRegularization, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.tau = tau
        self.alpha = alpha
        self.reg_coef = reg_coef

    def forward(self, input, target, gamma):
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction) + self.reg_coef * self.regularization_gumbel_softmax(gamma, self.tau, self.alpha)


class CrossEntropyLoss_DirichletRegularization(_WeightedLoss):
    def regularization_dirichlet(self, gamma, alpha):
   #     assert sum(gamma[0:len(gamma)]) == 1.
        K = len(gamma)
        tmp = gamma_func(K * alpha) / pow(gamma_func(alpha), K)

        for g in gamma:
         
            tmp *= g

        return abs(log(tmp, 2))

    def __init__(self, alpha,reg_coef):
        super(CrossEntropyLoss_DirichletRegularization, self).__init__()
        self.alpha = alpha
        self.reg_coef = reg_coef

    def forward(self, input, target,gamma):
        return F.cross_entropy(input, target) + self.reg_coef*self.regularization_dirichlet(gamma, self.alpha)