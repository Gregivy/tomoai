import tensorflow as tf

def sampler(img, points, mask = None):
  '''
  Performs bilinear sampling.
  If mask is present then sampling is done only for masked pixels.
  Sampling is based on https://arxiv.org/abs/1506.02025.
  Formula:
    output_ij = sum_n(sum_m(img_nm*x_coeff*y_coeff));
    x_coeff = max(0, 1 - |points_x_nm - i|);
    y_coeff = max(0, 1 - |points_y_nm - j|).

  Args:
  img - batch of image, tf.float32, [batchsize,height,width,channels].
  points - batch of new normalized coordinates [-1;1] for each pixel, tf.float32, [batchsize,height,width,2].
  mask - batch of masks, tf.float32, optional, [batchsize,height,width,1].

  Returns:
  tuple (warped_image, warped_mask)
    warped_image - sampled image [batchsize,height,width,channels], tf.float32.
    warped_mask - principled mask [batchsize,height,width,1], tf.float32.
  '''

  def multiply_and_reorder(a,b):
    '''
    Returns matrix multiplied 'a' on 'b' [batchsize, height, width, chnls].
    '''
    height, chnls, batchsize, points = a.shape
    width = b.shape[0]

    a = tf.transpose(a, [1,2,0,3])
    b = tf.transpose(b, [1,2,3,0])

    c = tf.linalg.matmul(a,b)
    c = tf.reshape(c, [chnls,batchsize,height,width])
    c = tf.transpose(c, [1,2,3,0])
    return c

  def flatten_and_reorder(img):
    '''
    Returns flattened and reordered 'img' [chnls, batchsize, height*width].
    '''
    batchsize, height, width, chnls = img.shape
    flatten_img = tf.reshape(img, [batchsize, height*width, chnls])
    flatten_img = tf.transpose(flatten_img, [2,0,1])
    return flatten_img

  def bilinear_coefficients(points,grid):
    '''
    Computes coefficients for 'points' on 'grid' [length, chnls, batchsize, height*width].
    '''
    length = grid.shape[-1]
    c = tf.expand_dims(points, -1)
    c = 1.0 - tf.abs(c - grid)
    c = tf.math.maximum(0.0, c)
    #c = 2/(1+tf.math.exp(-10*c))-1
    #c = 1/(1+tf.math.exp(-30*(c-0.2)))
    #c = 2*(1/(1+tf.math.exp(-30*c)) - 0.5)

    c = tf.transpose(c, [2,0,1])
    c = tf.expand_dims(c, 1)
    return c

  batchsize, height, width, chnls = img.shape

  # Scale indices from [-1;1] to [0, width/height - 1].
  points = (points + 1.0) * tf.constant([height - 1, width - 1], tf.float32) / 2

  # if not mask is None:
  #   points *= mask

  img_flatten = flatten_and_reorder(img)

  points_flatten = tf.reshape(points, [batchsize, height*width, 2])

  grid_x = tf.range(0, height, 1.0)
  grid_y = tf.range(0, width, 1.0)

  x_coeff = bilinear_coefficients(points_flatten[:,:,0], grid_x)
  y_coeff = bilinear_coefficients(points_flatten[:,:,1], grid_y)

  warped_image = multiply_and_reorder(img_flatten*x_coeff,y_coeff)

  warped_image = tf.math.minimum(1.0,warped_image)

  warped_mask = None
  if mask is None:
    warped_mask = multiply_and_reorder(x_coeff,y_coeff)
  else:
    mask_flatten = flatten_and_reorder(mask)
    warped_mask = multiply_and_reorder(mask_flatten*x_coeff,y_coeff)

  warped_mask = tf.math.minimum(1.0,warped_mask)
  # Differentiable binary thresholding
  warped_mask = 2/(1+tf.math.exp(-30*warped_mask))-1

  return warped_image, warped_mask
