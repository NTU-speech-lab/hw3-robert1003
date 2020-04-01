import torch
import torch.nn as nn

class Model0(nn.Module):

    def __init__(self, f1, f2, num_classes=11):
        super().__init__()
        self.f1 = f1
        self.f2 = f2
        self.classifier = nn.Sequential(
            nn.Linear(16384, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x1 = torch.flatten(self.f1(x), 1)
        x2 = torch.flatten(self.f2(x), 1)
        x = torch.cat((x1, x2), 1)
        x = self.classifier(x)
        return x

class Model1(nn.Module):
            
    def __init__(self, f1, f2, num_classes=11):
        super().__init__()
        self.f1 = f1
        self.f2 = f2
        self.classifier = nn.Sequential(
            nn.Linear(16384, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x1 = torch.flatten(self.f1(x), 1)
        x2 = torch.flatten(self.f2(x), 1)
        x = torch.cat((x1, x2), 1)
        x = self.classifier(x)
        return x

class Model2(nn.Module):
            
    def __init__(self, f1, f2, num_classes=11):
        super().__init__()
        self.f1 = f1
        self.f2 = f2
        self.classifier = nn.Sequential(
            nn.Linear(16384, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x1 = torch.flatten(self.f1(x), 1)
        x2 = torch.flatten(self.f2(x), 1)
        x = torch.cat((x1, x2), 1)
        x = self.classifier(x)
        return x

class Model3(nn.Module):
            
    def __init__(self, f1, f2, num_classes=11):
        super().__init__()
        self.f1 = f1
        self.f2 = f2
        self.classifier = nn.Sequential(
            nn.Linear(16384, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        x1 = torch.flatten(self.f1(x), 1)
        x2 = torch.flatten(self.f2(x), 1)
        x = torch.cat((x1, x2), 1)
        x = self.classifier(x)
        return x

class Model4(nn.Module):

    def __init__(self, features, num_classes=11):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Model5(nn.Module):

    def __init__(self, features, num_classes=11):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class Model6(nn.Module):

    def __init__(self, features, num_classes=11):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
