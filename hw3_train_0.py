#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[2]:


import cv2, torch, torchvision, time, os
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# In[3]:


# hyperparams
batch_size = 128
valid_ratio = 0.25


# In[4]:


# set random seed
np.random.seed(881003)
torch.manual_seed(890108)
torch.cuda.manual_seed(880301)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[5]:


# load data
X = np.vstack([np.load('./data/train_X.npy'), np.load('./data/valid_X.npy')])
y = np.hstack([np.load('./data/train_y.npy'), np.load('./data/valid_y.npy')])
test_X = np.load('./data/test_X.npy')
print(X.shape, y.shape, test_X.shape)


# In[6]:


def train_test_split(X, y, test_ratio=0.2, balanced=True):
    if not balanced:
        train_size = int(X.shape[0] * (1 - test_ratio) + 0.5)
        choice = np.random.permutation(X.shape[0])
        print(len(choice))
        return X[choice[:train_size]], X[choice[train_size:]], y[choice[:train_size]], y[choice[train_size:]]
    else:
        b = {}
        for k, z in enumerate(y):
            if z not in b:
                b[z] = []
            b[z].append(k)
        train_index, test_index = [], []
        for k in b:
            b[k] = np.array(b[k])
            np.random.shuffle(b[k])
            train_size = int(len(b[k]) * (1 - test_ratio) + 0.5)
            train_index.append(b[k][:train_size])
            test_index.append(b[k][train_size:])
        train_index = np.hstack(train_index)
        test_index = np.hstack(test_index)
        return X[train_index], X[test_index], y[train_index], y[test_index]        


# In[7]:


train_X, val_X, train_y, val_y = train_test_split(X, y, test_ratio=valid_ratio)
print(train_X.shape, train_y.shape, val_X.shape, val_y.shape, test_X.shape)


# In[8]:


num, cnt = np.unique(train_y, return_counts=True)
print(num, cnt)


# In[9]:


data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomCrop(96, padding_mode='symmetric'),
        transforms.RandomRotation(45),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        ], p=0.8),
        transforms.RandomApply([
            transforms.RandomGrayscale(p=1.0),
        ], p=0.3),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=30, translate=(0, 0.2), scale=(0.9, 1), fillcolor=0)
        ], p=0.8),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
}

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


# In[10]:


train_set = ImgDataset(train_X, train_y, transform=data_transforms['train'])
val_set = ImgDataset(val_X, val_y, transform=data_transforms['test'])
test_set = ImgDataset(test_X, transform=data_transforms['test'])

sampler = WeightedRandomSampler([1.0 for data, label in train_set], len(train_set), replacement=True)

train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# In[11]:


class Model(nn.Module):
            
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
            
def _make_layers(cfg, ):
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


# In[12]:


model = Model(
    _make_layers([
        32, 32, 32, 'M', 
        64, 64, 64, 'M', 
        128, 128, 128, 'M', 
        256, 256, 256, 256, 'M', 
        512, 512, 512, 512, 'EM'
    ]),
    _make_layers([
        32, 32, 32, 'A', 
        64, 64, 64, 'A', 
        128, 128, 128, 'A', 
        256, 256, 256, 256, 'A', 
        512, 512, 512, 512, 'EA'
    ])
).to(device)


# In[13]:


loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10000
patience = 50


# In[14]:


best_acc, best_loss = 0, np.inf


# In[15]:


cnt = 0

for epoch in range(epochs):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0


    for i, data in enumerate(train_loader):
        model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].to(device)) # 利用 model 得到預測的機率分佈 這邊實際ä¸就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].to(device)) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].to(device))
            batch_loss = loss(val_pred, data[1].to(device))

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        #將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % (epoch + 1, epochs, time.time()-epoch_start_time,              train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
        
        if val_acc/val_set.__len__() > best_acc:
            print(f'acc increase: {best_acc} -> {val_acc/val_set.__len__()}, saving model...')
            best_acc = val_acc/val_set.__len__()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, 'model0.ckpt')
            cnt = 0
        else:
            cnt += 1
            if cnt > patience:
                break
