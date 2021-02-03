import tensorflow as tf
from photoloss import photoloss
from utils import get_transform_mat, inverse_warp, gray2rgb
from flip_loss_tf import FLIPLoss
from radon import radon, sino2multi
from generator import norm, im_norm

flip = FLIPLoss()

def xentropy(outputs, real):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(outputs, real))

def cat_xentropy(outputs, real):
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(outputs, real))

def l2_norm_squared(x, axis = 1):
    return tf.reduce_sum(tf.math.square(x), axis)

def smooth_reg(x, y, da):
    ro = l2_norm_squared(x - y, 1)
    k = 4*tf.math.square(tf.math.sin(da/2))
    return 1 - 1/(1+ro/k)
    #return ro/k
    #k = 2*k
    #return 1 - tf.math.exp(-(ro/k))
    
def l1_loss(a, b, axis = -1):
    return tf.reduce_mean(tf.math.abs(a - b), axis)

def mse(true_val, pred_val, axis = -1):
    return tf.reduce_mean(tf.math.square(true_val - pred_val), axis)

def gradient_penalty(x, x_gen, model):
    epsilon = tf.random.uniform([x.shape[0], 1, 1, 1], 0.0, 1.0)
    x_hat = epsilon * x + (1 - epsilon) * x_gen
    with tf.GradientTape() as t:
        t.watch(x_hat)
        d_hat = model(x_hat)
    gradients = t.gradient(d_hat, x_hat)
    ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
    d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
    return d_regularizer

class Losses():
    def __init__(self, models):
        self.models = models
    
    def loss_f(self, sino, pov, im_shape):
        slice_net = self.models['slice_net']
        sino_multi = sino2multi(sino, im_shape)
        
        slices = slice_net(sino_multi)
        
        #_, h, w, _ = sino.shape
        _, h, w, _ = slices.shape
        
        #loss_r = tf.math.minimum(l1_loss(sino, norm(radon(slices, pov)), None), l1_loss(sino, norm(radon(1 - slices, pov)), None))
        
        #loss_r = l1_loss(sino_multi, sino2multi(norm(radon(slices, pov)), im_shape), None)
        gen_sino = norm(radon(slices, pov),h)
        #loss_r = 0.15*tf.reduce_mean((1-tf.image.ssim(sino, gen_sino, 1))/2) + 0.85*l1_loss(sino, gen_sino, None) # l1 and ssim
        loss_r = l1_loss(sino, gen_sino, None) # normal
        #loss_r = l1_loss(
        #    tf.image.resize(sino, [h*2, w*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), 
        #    tf.image.resize(norm(radon(slices, pov)), [h*2, w*2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), None) # normal
        #loss_r = mse(sino, norm(radon(slices, pov)), None) # mse
        #loss_r = flip(
        #    tf.transpose(gray2rgb(sino), perm=[0, 3, 1, 2]), 
        #    tf.transpose(gray2rgb(norm(radon(slices, pov))), perm=[0, 3, 1, 2])) # with flip
        #loss_r = xentropy(sino, norm(radon(slices, pov))) # crossentropy

        return loss_r
