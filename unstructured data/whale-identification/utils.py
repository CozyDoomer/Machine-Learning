import numpy as np
import torch
import pandas as pd

from fastai.vision import Image, open_image, resize_to, pil2tensor

import cv2

# adapted from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
def open_with_bb(path):
    ratio = 3
    sz0 = 192
    sz = (ratio*sz0,sz0)
    
    bb = pd.read_csv('data/bounding_boxes.csv')
    bb.set_index('Image', inplace=True)
    img = open_image(path, convert_mode='L').data
    
    fname = path.split('/')[-1]
    x0,y0,x1,y1 = tuple(bb.loc[fname,['x0','y0','x1','y1']].tolist())
    l1,l0,_ = img.shape
    b0,b1 = x1-x0 + 50, y1-y0 + 20 #add extra paddning
    b0n,b1n = (b0, b0/ratio) if b0**2/ratio > b1**2*ratio else (b1*ratio, b1)
    if b0n > l0: b0n,b1n = l0,b1n*l0/b0n
    if b1n > l1: b0n,b1n = b0n*l1/b1n,l1
    x0n = (x0 + x1 - b0n)/2
    x1n = (x0 + x1 + b0n)/2
    y0n = (y0 + y1 - b1n)/2
    y1n = (y0 + y1 + b1n)/2
    x0n,x1n,y0n,y1n = int(x0n),int(x1n),int(y0n),int(y1n)
    if(x0n < 0): x0n,x1n = 0,x1n-x0n
    elif(x1n > l0): x0n,x1n = x0n+l0-x1n,l0
    if(y0n < 0): y0n,y1n = 0,y1n-y0n
    elif(y1n > l1): y0n,y1n = y0n+l1-y1n,l1
        
    #print(img.shape)
    img = cv2.resize(img.numpy()[:,y0n:y1n,x0n:x1n], sz)
    
    return Image(pil2tensor(img[0,:,:], np.float32).float())

# https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def map5(preds, targs):
    predicted_idxs = preds.sort(descending=True)[1]
    top_5 = predicted_idxs[:, :5]
    res = mapk([[t] for t in targs.cpu().numpy()], top_5.cpu().numpy(), 5)
    return torch.tensor(res)

def top_5_preds(preds): return np.argsort(preds.numpy())[:, ::-1][:, :5]

def top_5_pred_labels(preds, classes):
    top_5 = top_5_preds(preds)
    labels = []
    for i in range(top_5.shape[0]):
        labels.append(' '.join([classes[idx] for idx in top_5[i]]))
    return labels

def create_submission(preds, data, name, classes=None):
    if not classes: classes = data.classes
    sub = pd.DataFrame({'Image': [path.name for path in data.test_ds.x.items]})
    sub['Id'] = top_5_pred_labels(preds, classes)
    sub.to_csv(f'subs/{name}.csv.gz', index=False, compression='gzip')


def intersection(preds, targs):
    # preds and targs are of shape (bs, 4), pascal_voc format
    max_xy = torch.min(preds[:, 2:], targs[:, 2:])
    min_xy = torch.max(preds[:, :2], targs[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]

def area(boxes):
    return ((boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1]))

def union(preds, targs):
    return area(preds) + area(targs) - intersection(preds, targs)

def IoU(preds, targs):
    return intersection(preds, targs) / union(preds, targs)