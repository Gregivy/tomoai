import tensorflow as tf
from utils import _meshgrid_abs, _pixel2cam, _cam2pixel, get_transform_mat, inverse_warp
from sampler import sampler

def _spatial_transformer_2(img, coords):
    """A wrapper over binlinear_sampler(), taking absolute coords as input."""
    img_height = tf.cast(tf.shape(img)[1], tf.float32)
    img_width = tf.cast(tf.shape(img)[2], tf.float32)
    px = coords[:, :, :, :1]
    py = coords[:, :, :, 1:]
    # Normalize coordinates to [-1, 1] to send to _bilinear_sampler.
    px = px / (img_width - 1) * 2.0 - 1.0
    py = py / (img_height - 1) * 2.0 - 1.0
    output_img, mask = sampler(img, tf.stack([py,px], -1))
    return output_img, mask

def warp(img, depth, egomotion_mat, intrinsic_mat,
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
    projected_img, mask = _spatial_transformer_2(img, source_pixel_coords)
    if flow:
        rigid_flow = source_pixel_coords - target_pixel_coords
        return projected_img, rigid_flow
    return projected_img, mask

def inference(models, img, intr, rt):
    
    #img_for_warp = tf.concat([img, models['feature_net'](img)], -1)
    img_for_warp = models['feature_net'](img)
    depth = 1 / models['depth_net'](img)[0]
    
    egomotion_mat_i_j_1 = get_transform_mat(rt, 1, 0)
    egomotion_mat_i_j_2 = get_transform_mat(rt, 0, 1)

#     warped_image_1, warp_mask_1 = (
#         warp(
#             img_for_warp,
#             depth,
#             egomotion_mat_i_j_1,
#             intr[:, 0, :, :],
#             tf.linalg.inv(intr[:, 0, :, :])
#         )
#     )
    
    warped_image_2, warp_mask_2 = (
        inverse_warp(
            img_for_warp,
            depth,
            egomotion_mat_i_j_2,
            intr[:, 0, :, :],
            tf.linalg.inv(intr[:, 0, :, :])
        )
    )
    
    
    #si_1 = warped_image_1 * warp_mask_1
    si_2 = warped_image_2 * warp_mask_2
    
    return si_2, models['refinement_net'](si_2)