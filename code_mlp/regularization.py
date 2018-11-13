from math import pow, factorial, log
from math import gamma as gamma_func

import torch
from torch.nn import Module
import torch.nn.functional as F


class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        

class CrossEntropyLoss_GumbelSoftmaxRegularization(_WeightedLoss):
    def regularization_gumbel_softmax(gamma, temperature, alpha):
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

    def __init__(self, tau, alpha, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='elementwise_mean'):
        super(CrossEntropyLoss_GumbelSoftmaxRegularization, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.tau = tau
        self.alpha = alpha

    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction) + regularization_gumbel_softmax(input, self.tau, self.alpha)


class CrossEntropyLoss_DirichletRegularization(_WeightedLoss):
    def regularization_dirichlet(gamma, alpha):
        assert sum(gamma[0:len(gamma)]) == 1.0

        K = len(gamma)
        tmp = gamma_func(K * alpha) / pow(gamma_func(alpha), K)

        for i in range(K):
            tmp *= gamma[i]

        return log(tmp, 2)

    def __init__(self, alpha, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='elementwise_mean'):
        super(CrossEntropyLoss_DirichletRegularization, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.alpha = alpha

    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction) + regularization_dirichlet(input, self.alpha)

