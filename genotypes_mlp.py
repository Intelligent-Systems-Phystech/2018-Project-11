# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 20:54:52 2018

@author: Valeriy
"""

from collections import namedtuple

Genotype = namedtuple('Genotype', 'structure')

PRIMITIVES = [
    'tanh',
    'relu',
    'softmax',
]