import torch
import torch.nn as nn

def _make_layers0(cfg, ):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'EM':
            layers += [nn.AdaptiveMaxPool2d((4, 4))]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        elif v == 'EA':
            layers += [nn.AdaptiveAvgPool2d((4, 4))]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _make_layers1(cfg, ):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'EM':
            layers += [nn.AdaptiveMaxPool2d((4, 4))]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        elif v == 'EA':
            layers += [nn.AdaptiveAvgPool2d((4, 4))]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _make_layers2(cfg, ):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'EM':
            layers += [nn.AdaptiveMaxPool2d((4, 4))]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        elif v == 'EA':
            layers += [nn.AdaptiveAvgPool2d((4, 4))]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _make_layers3(cfg, ):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'EM':
            layers += [nn.AdaptiveMaxPool2d((4, 4))]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        elif v == 'EA':
            layers += [nn.AdaptiveAvgPool2d((4, 4))]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _make_layers4(cfg, ):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _make_layers5(cfg, ):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _make_layers6(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
