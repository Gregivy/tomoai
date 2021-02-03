import math
import tensorflow as tf
import tensorflow_addons as tfa

def radon(img, ang):
    batch_size, height, width, channels = img.shape
    #img = tf.expand_dims(img, -1)
    images = tf.tile(tf.expand_dims(img, 1), [1, ang, 1, 1, 1]) # b a h w c
    images = tf.reshape(images, [batch_size * ang, height, width, channels]) # b*a h w c
    
    angles = tf.range(0, math.pi, math.pi / ang)
    angles = tf.tile(angles, [batch_size])
    
    rotated_img = tfa.image.rotate(
        images = images,
        angles = -angles,
        interpolation = 'bilinear',
        fill_mode = 'constant',
        #fill_value = 0.0
    )
    
    rotated_img = tf.reshape(rotated_img, [batch_size, ang, height, width, channels]) # b a h w c
    sino = tf.reduce_sum(rotated_img, axis=2) # b a w c
    return sino

def sino2multi(sino, img_shape):
    ang = sino.shape[1]
    batch_size, height, width, channels = img_shape
    
    angles = tf.range(0, math.pi, math.pi / ang)
    angles = tf.tile(angles, [batch_size])
    
    multi_img = tf.expand_dims(sino, 2)
    multi_img = tf.tile(multi_img, [1, 1, height, 1, 1]) # b a h w c
    multi_img = tf.reshape(multi_img, [batch_size*ang, height, width, channels]) # b*a h w c
    
    rotated_multi_img = tfa.image.rotate(
        images = multi_img,
        angles = angles,
        interpolation = 'bilinear',
        fill_mode = 'nearest',
        #fill_value = 0.0
    )
    
    rotated_multi_img = tf.reshape(rotated_multi_img, [batch_size, ang, height, width, channels]) # b a h w c
    
    final_multi_img = tf.transpose(rotated_multi_img, [0, 2, 3, 1, 4]) # b h w a c
    final_multi_img = tf.reshape(final_multi_img, [batch_size, height, width, channels * ang]) # b h w a*c
    
    return final_multi_img