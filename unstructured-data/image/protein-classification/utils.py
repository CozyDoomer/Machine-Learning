
import PIL
import numpy as np

from fastai.vision.image import *
from fastai.vision import Image


# adapted from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
def open_4_channel(fname):
    fname = str(fname)
    # strip extension before adding color
    if fname.endswith('.png'):
        fname = fname[:-4]
    colors = ['red','green','blue','yellow']
    
    img = [open_image(fname+'_'+color+'.png', convert_mode='L').data for color in colors]
    
    x = np.stack(img, axis=-1)
    
    return Image(pil2tensor(x[0,:,:], np.float32).float())