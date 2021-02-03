import tensorflow as tf
import math
import numpy as np

def gray2rgb(img):
    return tf.tile(img, [1, 1, 1, 3])

def degreesToRadians(d):
    return d * math.pi / 180

def radiansToDegrees(d):
    return d * 180 / math.pi

def angleNormPair(d):
    return tf.stack([tf.math.sin(d),tf.math.cos(d)],-1)

def angleNorm(a):
    return degreesToRadians(tf.cast(a, dtype=tf.float32)) / (math.pi * 2)

def angleDeNorm(a):
    return radiansToDegrees(a * (math.pi * 2))

def resize_like(inputs, ref):
    i_h, i_w = inputs.get_shape()[1], inputs.get_shape()[2]
    r_h, r_w = ref.get_shape()[1], ref.get_shape()[2]
    if i_h == r_h and i_w == r_w:
        return inputs
    else:
        # TODO(casser): Other interpolation methods could be explored here.
        return tf.image.resize(inputs, [r_h, r_w], method=tf.image.ResizeMethod.BILINEAR)
    
def make_intrinsics_matrix(intrinsics_list):
    fx, fy, cx, cy = list(np.array(intrinsics_list, dtype=np.float32))
    r1 = tf.stack([fx, 0, cx])
    r2 = tf.stack([0, fy, cy])
    r3 = tf.constant([0., 0., 1.], dtype=tf.float32)
    intrinsics = tf.stack([r1, r2, r3])
    return intrinsics

def get_multi_scale_intrinsics(intrinsics, num_scales = 3):
    """Returns multiple intrinsic matrices for different scales."""
    intrinsics_multi_scale = []
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        fx = intrinsics[0, 0] / (2**s)
        fy = intrinsics[1, 1] / (2**s)
        cx = intrinsics[0, 2] / (2**s)
        cy = intrinsics[1, 2] / (2**s)
        intrinsics_multi_scale.append(make_intrinsics_matrix([fx, fy, cx, cy]))
    intrinsics_multi_scale = tf.stack(intrinsics_multi_scale)

    return intrinsics_multi_scale

def inverse_warp(img, depth, egomotion_mat, intrinsic_mat,
                 intrinsic_mat_inv, flow = False):
    """Inverse warp a source image to the target image plane.
    Args:
    img: The source image (to sample pixels from) -- [B, H, W, 3].
    depth: Depth map of the target image -- [B, H, W].
    egomotion_mat: Matrix defining egomotion transform -- [B, 4, 4].
    intrinsic_mat: Camera intrinsic matrix -- [B, 3, 3].
    intrinsic_mat_inv: Inverse of the intrinsic matrix -- [B, 3, 3].
    Returns:
    Projected source image
    """
    dims = tf.shape(img)
    batch_size, img_height, img_width = dims[0], dims[1], dims[2]
    depth = tf.reshape(depth, [batch_size, 1, img_height * img_width])
    grid = _meshgrid_abs(img_height, img_width)
    grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])
    if flow:
        target_pixel_coords = tf.transpose(grid, perm=[0, 2, 3, 1])
    cam_coords = _pixel2cam(depth, grid, intrinsic_mat_inv)
    ones = tf.ones([batch_size, 1, img_height * img_width])
    cam_coords_hom = tf.concat([cam_coords, ones], axis=1)

    # Get projection matrix for target camera frame to source pixel frame
    hom_filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    hom_filler = tf.tile(hom_filler, [batch_size, 1, 1])
    intrinsic_mat_hom = tf.concat(
        [intrinsic_mat, tf.zeros([batch_size, 3, 1])], axis=2)
    intrinsic_mat_hom = tf.concat([intrinsic_mat_hom, hom_filler], axis=1)
    proj_target_cam_to_source_pixel = tf.matmul(intrinsic_mat_hom, egomotion_mat)
    source_pixel_coords = _cam2pixel(cam_coords_hom,
                                   proj_target_cam_to_source_pixel)
    source_pixel_coords = tf.reshape(source_pixel_coords,
                                   [batch_size, 2, img_height, img_width])
    source_pixel_coords = tf.transpose(source_pixel_coords, perm=[0, 2, 3, 1])
    projected_img, mask = _spatial_transformer(img, source_pixel_coords)
    if flow:
        rigid_flow = source_pixel_coords - target_pixel_coords
        return projected_img, rigid_flow
    return projected_img, mask


def get_transform_mat(egomotion_vecs, i, j):
    """Returns a transform matrix defining the transform from frame i to j."""
    egomotion_transforms = []
    batchsize = tf.shape(egomotion_vecs)[0]
    if i == j:
        return tf.tile(tf.expand_dims(tf.eye(4, 4), axis=0), [batchsize, 1, 1])
    for k in range(min(i, j), max(i, j)):
        transform_matrix = _egomotion_vec2mat(egomotion_vecs[:, k, :], batchsize)
        #print('transform_matrix', transform_matrix)
        #transform_matrix = tf.where(tf.math.is_nan(transform_matrix), tf.zeros_like(transform_matrix), transform_matrix)
        if i > j:  # Going back in sequence, need to invert egomotion.
            try:
                egomotion_transforms.insert(0, tf.linalg.inv(transform_matrix))
            except Exception:
                print('uninvertable')
        else:  # Going forward in sequence
            egomotion_transforms.append(transform_matrix)

    # Multiply all matrices.
    egomotion_mat = egomotion_transforms[0]
    for i in range(1, len(egomotion_transforms)):
        egomotion_mat = tf.matmul(egomotion_mat, egomotion_transforms[i])

    return egomotion_mat


def _pixel2cam(depth, pixel_coords, intrinsic_mat_inv):
    """Transform coordinates in the pixel frame to the camera frame."""
    cam_coords = tf.matmul(intrinsic_mat_inv, pixel_coords) * depth
    return cam_coords


def _cam2pixel(cam_coords, proj_c2p):
    """Transform coordinates in the camera frame to the pixel frame."""
    pcoords = tf.matmul(proj_c2p, cam_coords)
    x = tf.slice(pcoords, [0, 0, 0], [-1, 1, -1])
    y = tf.slice(pcoords, [0, 1, 0], [-1, 1, -1])
    z = tf.slice(pcoords, [0, 2, 0], [-1, 1, -1])
    # Not tested if adding a small number is necessary
    x_norm = x / (z + 1e-10)
    y_norm = y / (z + 1e-10)
    pixel_coords = tf.concat([x_norm, y_norm], axis=1)
    return pixel_coords


def _meshgrid_abs(height, width):
    """Meshgrid in the absolute coordinates."""
    x_t = tf.matmul(
        tf.ones(shape=tf.stack([height, 1])),
        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(
        tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
        tf.ones(shape=tf.stack([1, width])))
    x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))
    ones = tf.ones_like(x_t_flat)
    grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)
    return grid


def _euler2mat(z, y, x):
    """Converts euler angles to rotation matrix.
    From:
    https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    TODO: Remove the dimension for 'N' (deprecated for converting all source
    poses altogether).
    Args:
    z: rotation angle along z axis (in radians) -- size = [B, n]
    y: rotation angle along y axis (in radians) -- size = [B, n]
    x: rotation angle along x axis (in radians) -- size = [B, n]
    Returns:
    Rotation matrix corresponding to the euler angles, with shape [B, n, 3, 3].
    """
    batch_size = tf.shape(z)[0]
    n = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([batch_size, n, 1, 1])
    ones = tf.ones([batch_size, n, 1, 1])

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz, cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny, zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    return tf.matmul(tf.matmul(xmat, ymat), zmat)


def _egomotion_vec2mat(vec, batch_size):
    """Converts 6DoF transform vector to transformation matrix.
    Args:
    vec: 6DoF parameters [tx, ty, tz, rx, ry, rz] -- [B, 6].
    batch_size: Batch size.
    Returns:
    A transformation matrix -- [B, 4, 4].
    """
    translation = tf.slice(vec, [0, 0], [-1, 3])
    translation = tf.expand_dims(translation, -1)
    rx = tf.slice(vec, [0, 3], [-1, 1])
    ry = tf.slice(vec, [0, 4], [-1, 1])
    rz = tf.slice(vec, [0, 5], [-1, 1])
    rot_mat = _euler2mat(rz, ry, rx)
    rot_mat = tf.squeeze(rot_mat, axis=[1])
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat


def _bilinear_sampler(im, x, y, name='blinear_sampler'):
    """Perform bilinear sampling on im given list of x, y coordinates.
    Implements the differentiable sampling mechanism with bilinear kernel
    in https://arxiv.org/abs/1506.02025.
    x,y are tensors specifying normalized coordinates [-1, 1] to be sampled on im.
    For example, (-1, -1) in (x, y) corresponds to pixel location (0, 0) in im,
    and (1, 1) in (x, y) corresponds to the bottom right pixel in im.
    Args:
    im: Batch of images with shape [B, h, w, channels].
    x: Tensor of normalized x coordinates in [-1, 1], with shape [B, h, w, 1].
    y: Tensor of normalized y coordinates in [-1, 1], with shape [B, h, w, 1].
    name: Name scope for ops.
    Returns:
    Sampled image with shape [B, h, w, channels].
    Principled mask with shape [B, h, w, 1], dtype:float32.  A value of 1.0
      in the mask indicates that the corresponding coordinate in the sampled
      image is valid.
    """
    #with tf.variable_scope(name):
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])

    # Constants.
    batch_size = tf.shape(im)[0]
    _, height, width, channels = im.get_shape().as_list()

    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')
    zero = tf.constant(0, dtype=tf.int32)
    max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
    max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

    # Scale indices from [-1, 1] to [0, width - 1] or [0, height - 1].
    x = (x + 1.0) * (width_f - 1.0) / 2.0
    y = (y + 1.0) * (height_f - 1.0) / 2.0

    # Compute the coordinates of the 4 pixels to sample from.
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    mask = tf.logical_and(
        tf.logical_and(x0 >= zero, x1 <= max_x),
        tf.logical_and(y0 >= zero, y1 <= max_y))
    mask = tf.cast(mask, 'float32')

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    dim2 = width
    dim1 = width * height

    # Create base index.
    base = tf.range(batch_size) * dim1
    base = tf.reshape(base, [-1, 1])
    base = tf.tile(base, [1, height * width])
    base = tf.reshape(base, [-1])

    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # Use indices to lookup pixels in the flat image and restore channels dim.
    im_flat = tf.reshape(im, tf.stack([-1, channels]))
    im_flat = tf.cast(im_flat, 'float32')
    pixel_a = tf.gather(im_flat, idx_a)
    pixel_b = tf.gather(im_flat, idx_b)
    pixel_c = tf.gather(im_flat, idx_c)
    pixel_d = tf.gather(im_flat, idx_d)

    x1_f = tf.cast(x1, 'float32')
    y1_f = tf.cast(y1, 'float32')

    # And finally calculate interpolated values.
    wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
    wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
    wc = tf.expand_dims(((1.0 - (x1_f - x)) * (y1_f - y)), 1)
    wd = tf.expand_dims(((1.0 - (x1_f - x)) * (1.0 - (y1_f - y))), 1)

    output = tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
    output = tf.reshape(output, tf.stack([batch_size, height, width, channels]))
    mask = tf.reshape(mask, tf.stack([batch_size, height, width, 1]))
    return output, mask

def _spatial_transformer(img, coords):
    """A wrapper over binlinear_sampler(), taking absolute coords as input."""
    img_height = tf.cast(tf.shape(img)[1], tf.float32)
    img_width = tf.cast(tf.shape(img)[2], tf.float32)
    px = coords[:, :, :, :1]
    py = coords[:, :, :, 1:]
    # Normalize coordinates to [-1, 1] to send to _bilinear_sampler.
    px = px / (img_width - 1) * 2.0 - 1.0
    py = py / (img_height - 1) * 2.0 - 1.0
    output_img, mask = _bilinear_sampler(img, px, py)
    return output_img, mask

def get_seq_middle(seq_length):
    """Returns relative index for the middle frame in sequence."""
    half_offset = int((seq_length - 1) / 2)
    return seq_length - 1 - half_offset

def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]

def depth_smoothness(depth, img):
    """Computes image-aware depth smoothness loss."""
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_dx), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_dy), 3, keepdims=True))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))

def ssim(x, y):
    """Computes a differentiable structured image similarity measure."""
    c1 = 0.01**2  # As defined in SSIM to stabilize div. by small denominator.
    c2 = 0.03**2
    mu_x = tf.nn.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = tf.nn.avg_pool2d(y, 3, 1, 'VALID')
    sigma_x = tf.nn.avg_pool2d(tf.pow(x,2), 3, 1, 'VALID') - tf.pow(mu_x,2)
    sigma_y = tf.nn.avg_pool2d(tf.pow(y,2), 3, 1, 'VALID') - tf.pow(mu_y,2)
    sigma_xy = tf.nn.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    ssim_d = (tf.pow(mu_x,2) + tf.pow(mu_y,2) + c1) * (sigma_x + sigma_y + c2)
    ssim = ssim_n / ssim_d
    return tf.clip_by_value((1 - ssim) / 2, 0, 1)
