import tensorflow as tf
import utils
from PIL import Image, ImageOps
#import cv2 as cv
from random import shuffle
from glob import glob
import numpy as np
import tomopy

def custom_preprocess_image(im,h,w,chns):
    loadedim = im.resize((h,w))
    if chns == 1:
        loadedim = loadedim.convert('L')
        #loadedim = ImageOps.invert(loadedim)
    elif chns == 3:
        loadedim = loadedim.convert('RGB')
    im_frame = np.array(loadedim).reshape(h,w,chns)
    im_frame = tf.image.convert_image_dtype(im_frame, tf.float32)
    
    return im_frame

def im_norm(a):
    a_max = tf.reduce_max(a)
    
    return a/a_max

def norm(a, h):
    return a / h

class Generator(tf.keras.utils.Sequence):
    def __init__(
        self, 
        h, w, chns, 
        batch_size, pov):
        
        obj = tomopy.shepp3d(size=h)
        #print(tf.reduce_max(tf.constant(obj)))
        ang = tomopy.angles(pov)
        sino = tomopy.project(obj, ang, pad = False)
        #print(tf.reduce_max(tf.constant(sino)))
        rec = tomopy.recon(sino, ang, algorithm='art', num_gridx=w, num_gridy=h)
        #print(tf.reduce_max(tf.constant(rec)))
        
        self.batch_size = batch_size
        self.pov = pov
        
        self.obj = tf.expand_dims(tf.cast(obj, tf.float32), -1)
        #self.obj = self.obj / 1.0
        
        self.sino = tf.expand_dims(tf.cast(sino, tf.float32), -1)
        
        self.rec = im_norm(tf.expand_dims(tf.cast(rec, tf.float32), -1))
        #self.rec = self.rec / 1.0
        
        print(self.obj.shape, self.sino.shape, self.rec.shape)
        
        self.train_images = []
        
        for slice_i in range(self.obj.shape[-2]):
            self.train_images.append([
                self.obj[slice_i], 
                norm(self.sino[:,slice_i,:], h), 
                self.rec[slice_i]
            ])
        
        self.shuffleTrain()

    def __len__(self):
        subset = self.train_images
        return len(subset) // self.batch_size

    def __getitem__(self, i):
        subset = self.train_images
        b = subset[i*self.batch_size:i*self.batch_size+self.batch_size]
        
        gt = tf.convert_to_tensor([p[0] for p in b])
        sino = tf.convert_to_tensor([p[1] for p in b])
        rec = tf.convert_to_tensor([p[2] for p in b])

        return {
            'gt': gt,
            'sino': sino,
            'rec': rec,
            'pov': self.pov
        }
        
    def shuffleTrain(self):
        shuffle(self.train_images)

    def on_epoch_end(self):
        self.shuffleTrain()