# setup environment
#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

# import packages
import torch, torchvision, os, sys
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from _utils import readfile
from _dataset import ImgDataset
from _model import m0, m1, m2, m3, m4, m5, m6

# define hyperparams
data_dir = sys.argv[1]
pred_path = sys.argv[2]
batch_size = 64
pic_size = 128
top_num = 2

model_idx = [0, 1, 2, 3, 4, 5, 6]
model_path = {
    0:'./model0.ckpt',
    1:'./model1.ckpt',
    2:'./model2.ckpt',
    3:'./model3.ckpt',
    4:'./model4.ckpt',
    5:'./model5.ckpt',
    6:'./model6.ckpt'
}
device = 'cuda'

# load test data
test_X = readfile(path=os.path.join(data_dir, 'testing'), label=False, pic_size=pic_size)

print('test data loaded! test data shape={}'.format(test_X.shape))

# convert it into Dataset and DataLoader
test_set = ImgDataset(test_X, transform=transforms.Compose([ transforms.ToPILImage(), transforms.ToTensor() ]))
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# load model

def get_model(idx):
    if idx == 0:
        return m0()
    elif idx == 1:
        return m1()
    elif idx == 2:
        return m2()
    elif idx == 3:
        return m3()
    elif idx == 4:
        return m4()
    elif idx == 5:
        return m5()
    elif idx == 6:
        return m6()

# predict
print('start predicting...')
predictions = []
for idx in model_idx:
    print('model {}'.format(idx))
    model = get_model(idx).to(device)
    checkpoint = torch.load(model_path[idx])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    prediction = []
    for i, x in enumerate(test_loader):
        x = x.to(device)
        prob = F.softmax(model(x).cpu().data, dim=1)
        prediction.append(prob)
        print('{}/{}'.format(i + 1, len(test_loader)), end='\r')
    predictions.append(torch.cat(prediction, dim=0))

    del model
    torch.cuda.empty_cache()
    print('done        ')
predictions = torch.stack(predictions)
cnt = torch.zeros((predictions.size(1), 11))
for prediction in predictions:
    for i, j in enumerate(prediction):
        cnt[i] += j
_, final_predict = torch.topk(cnt, 1, axis=1)
final_predict = final_predict.squeeze(1)

# write to output
with open(pred_path, 'w') as f:
    print('Id,Category', file=f)
    for i, y in enumerate(final_predict):
        print('{},{}'.format(i, y), file=f)
