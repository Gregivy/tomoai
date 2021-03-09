import math
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

def pad_image(img):
    _, h, w, _ = img.shape

    d = math.sqrt(h*h+w*w)
    ph = int((d - h)/2) + 1
    pw = int((d - w)/2) + 1
    img = tf.pad(img, [[0,0], [ph,ph], [pw,pw], [0,0]], 'CONSTANT')

    return img

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

def ramp_filter(size):
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                        np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    #print(f)

    # Computing the ramp filter from the fourier transform of its
    # frequency domain representation lessens artifacts and removes a
    # small bias as explained in [1], Chap 3. Equation 61
    fourier_filter = tf.cast(2 * tf.math.real(tf.signal.fft(f)), tf.complex64)      # ramp filters

    return fourier_filter

def apply_ramp_to_batch(batch):
    batch = tf.cast(batch, tf.complex64)

    b, size = batch.shape

    fourier_filter = ramp_filter(size)
    fourier_filter = tf.tile(tf.expand_dims(fourier_filter, 0), [b,1])

    return tf.math.real(tf.signal.ifft(tf.signal.fft(batch) * fourier_filter))

def apply_ramp_to_img(img):
    img = tf.cast(img, tf.complex64)

    b, h, w, c = img.shape
    size = h*w*c
    img = tf.reshape(img, [b, size])
    filtered_img = tf.reshape(apply_ramp_to_batch(img), [b,h,w,c])

    return filtered_img

def inv_radon(sino, img_shape):
    sm = sino2multi(sino, img_shape)

    return tf.reduce_mean(sm, -1, keepdims=True)
