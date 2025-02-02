import os, cv2
import numpy as np

def readfile(path, label, pic_size, channel_cnt=3):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), pic_size, pic_size, channel_cnt), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(pic_size, pic_size))
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x
        
