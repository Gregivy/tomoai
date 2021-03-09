import tensorflow as tf

def im_normalize(im, min_i, max_i):
    return (im - min_i)/(max_i - min_i)

def preprocess(im, h, w, min_i = 0.0, max_i = 255.0):
    im = tf.image.resize(tf.cast(im, tf.float32), [h, w])
    im = im_normalize(im, min_i, max_i)
    return im
