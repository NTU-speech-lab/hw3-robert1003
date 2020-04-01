import torch
import torch.nn
from _models import Model0, Model1, Model2, Model3, Model4, Model5, Model6
from _make_layers import _make_layers0, _make_layers1, _make_layers2, _make_layers3, _make_layers4, _make_layers5, _make_layers6

def m0():
    m0 = Model0(
        _make_layers0([
            32, 32, 32, 'M', 
            64, 64, 64, 'M', 
            128, 128, 128, 'M', 
            256, 256, 256, 256, 'M', 
            512, 512, 512, 512, 'EM'
        ]),
        _make_layers0([
            32, 32, 32, 'A', 
            64, 64, 64, 'A', 
            128, 128, 128, 'A', 
            256, 256, 256, 256, 'A', 
            512, 512, 512, 512, 'EA'
        ])
    )
    return m0

def m1():
    m1 = Model1(
        _make_layers1([
            32, 32, 32, 'M', 
            64, 64, 64, 'M', 
            128, 128, 128, 'M', 
            256, 256, 256, 256, 'M', 
            512, 512, 512, 512, 'EM'
        ]),
        _make_layers1([
            32, 32, 32, 'A', 
            64, 64, 64, 'A', 
            128, 128, 128, 'A', 
            256, 256, 256, 256, 'A', 
            512, 512, 512, 512, 'EA'
        ])
    )
    return m1

def m2():
    m2 = Model2(
        _make_layers2([
            32, 32, 32, 'M', 
            64, 64, 64, 'M', 
            128, 128, 128, 'M', 
            256, 256, 256, 256, 'M', 
            512, 512, 512, 512, 'EM'
        ]),
        _make_layers2([
            32, 32, 32, 'A', 
            64, 64, 64, 'A', 
            128, 128, 128, 'A', 
            256, 256, 256, 256, 'A', 
            512, 512, 512, 512, 'EA'
        ])
    )
    return m2

def m3():
    m3 = Model3(
        _make_layers3([
            32, 32, 32, 'M', 
            64, 64, 64, 'M', 
            128, 128, 128, 'M', 
            256, 256, 256, 256, 'M', 
            512, 512, 512, 512, 'EM'
        ]),
        _make_layers3([
            32, 32, 32, 'A', 
            64, 64, 64, 'A', 
            128, 128, 128, 'A', 
            256, 256, 256, 256, 'A', 
            512, 512, 512, 512, 'EA'
        ])
    )
    return m3

def m4():
    m4 = Model4(
        _make_layers4([
            64, 64, 'M', 
            128, 128, 'M',
            256, 256, 256, 256, 'M',
            512, 512, 512, 512, 'M',
            512, 512, 512, 512, 'M'
        ])
    )
    return m4

def m5():
    m5 = Model5(
        _make_layers5([
            64, 64, 'M', 
            128, 128, 'M',
            256, 256, 256, 256, 'M',
            512, 512, 512, 512, 'M',
            512, 512, 512, 512, 'M'
        ])
    )
    return m5

def m6():
    m6 = Model6(
        _make_layers6([
            32, 32, 32, 'M', 
            64, 64, 64, 'M', 
            128, 128, 128, 'M', 
            256, 256, 256, 256, 'M', 
            512, 512, 512, 512, 'M'
        ])
    )
    return m6
