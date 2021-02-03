import tensorflow as tf
import numpy as np
from utils import *

def photoloss(
    image_stack,
    disp,
    egomotion,
    intrinsic_mat,
    intrinsic_mat_inv,
    seq_length = 2,
    smooth_weight = 0.05,
    reconstr_weight = 0.85,
    ssim_weight = 0.15,
    compute_minimum_loss = False,
    depth_normalization = True,
    depth_upsampling = True,
    equal_weighting = True,
    exhaustive_mode = False,
    NUM_SCALES = 3,
    rec_loss = None
):
    """Computes loss
    image_stack a list with seq_length elements of [B, H, W, img_channels]
    """

    img_height, img_width, img_channels = image_stack[0].shape[1:4]

    reconstr_loss = 0
    smooth_loss = 0
    ssim_loss = 0
    
    #print(disp[0])

    image_stack = tf.concat(image_stack, axis=-1)

    # self.images is organized by ...[scale][B, h, w, seq_len * img_channels].
    images = [None for _ in range(NUM_SCALES)]
    # Following nested lists are organized by ...[scale][source-target].
    warped_image = [{} for _ in range(NUM_SCALES)]
    warp_mask = [{} for _ in range(NUM_SCALES)]
    warp_error = [{} for _ in range(NUM_SCALES)]
    ssim_error = [{} for _ in range(NUM_SCALES)]

    middle_frame_index = get_seq_middle(seq_length)

    for s in range(NUM_SCALES):
        # Scale image stack.
        if s == 0:  # Just as a precaution. TF often has interpolation bugs.
            images[s] = image_stack
        else:
            height_s = int(img_height / (2**s))
            width_s = int(img_width / (2**s))
            images[s] = tf.image.resize(
                image_stack,
                [height_s, width_s],
                method=tf.image.ResizeMethod.BILINEAR,
                preserve_aspect_ratio=True,
                antialias=True)

        # Smoothness.
        if smooth_weight > 0:
            for i in range(seq_length):
                # When computing minimum loss, use the depth map from the middle
                # frame only.
                if not compute_minimum_loss or i == middle_frame_index:
                    disp_smoothing = disp[i][s]
                    disp_input = None
                    if depth_normalization:
                        # Perform depth normalization, dividing by the mean.
                        mean_disp = tf.reduce_mean(disp_smoothing, axis=[1, 2, 3], keepdims=True)
                        disp_input = disp_smoothing / mean_disp
                    else:
                        disp_input = disp_smoothing
                    scaling_f = (1.0 if equal_weighting else 1.0 / (2**s))
                    smooth_loss += scaling_f * depth_smoothness(
                        disp_input, images[s][:, :, :, img_channels * i:img_channels * (i + 1)])

        debug_all_warped_image_batches = []
        for i in range(seq_length):
            for j in range(seq_length):
                if i == j:
                    continue

                # When computing minimum loss, only consider the middle frame as
                # target.
                if compute_minimum_loss and j != middle_frame_index:
                    continue
                # We only consider adjacent frames, unless either
                # compute_minimum_loss is on (where the middle frame is matched with
                # all other frames) or exhaustive_mode is on (where all frames are
                # matched with each other).
                if (not compute_minimum_loss and not exhaustive_mode and abs(i - j) != 1):
                    continue

                selected_scale = 0 if depth_upsampling else s
                source = images[selected_scale][:, :, :, img_channels * i:img_channels * (i + 1)]
                target = images[selected_scale][:, :, :, img_channels * j:img_channels * (j + 1)]

                target_depth = None

                if depth_upsampling:
                    target_depth = tf.image.resize(
                                    1 / disp[j][s],
                                    [img_height, img_width],
                                    method=tf.image.ResizeMethod.BILINEAR,
                                    preserve_aspect_ratio=True,
                                    antialias=True)
                else:
                    target_depth = 1 / disp[j][s]

                key = '%d-%d' % (i, j)

                # Don't handle motion, classic model formulation.
                egomotion_mat_i_j = get_transform_mat(egomotion, i, j)
                # Inverse warp the source image to the target image frame for
                # photometric consistency loss.
                warped_image[s][key], warp_mask[s][key] = (
                    inverse_warp(
                        source,
                        target_depth,
                        egomotion_mat_i_j,
                        intrinsic_mat[:, selected_scale, :, :],
                        intrinsic_mat_inv[:, selected_scale, :, :]))

                # Reconstruction loss.
                warp_error[s][key] = tf.abs(warped_image[s][key] - target)
                if not compute_minimum_loss:
                    if rec_loss:
                        reconstr_loss += rec_loss(
                            tf.transpose(warped_image[s][key] * warp_mask[s][key], perm=[0, 3, 1, 2]), 
                            tf.transpose(target * warp_mask[s][key], perm=[0, 3, 1, 2])
                        )
                    else:
                        reconstr_loss += tf.reduce_mean(
                            warp_error[s][key] * warp_mask[s][key])
                # SSIM.
                if ssim_weight > 0:
                    ssim_error[s][key] = ssim(warped_image[s][key], target)
                    # TODO(rezama): This should be min_pool2d().
                    if not compute_minimum_loss:
                        ssim_mask = tf.nn.avg_pool2d(warp_mask[s][key], 3, 1, 'VALID')
                        ssim_loss += tf.reduce_mean(ssim_error[s][key] * ssim_mask)

        # If the minimum loss should be computed, the loss calculation has been
        # postponed until here.
        if compute_minimum_loss:
            for frame_index in range(middle_frame_index):
                key1 = '%d-%d' % (frame_index, middle_frame_index)
                key2 = '%d-%d' % (seq_length - frame_index - 1,
                                  middle_frame_index)
                min_error = tf.minimum(warp_error[s][key1],
                                       warp_error[s][key2])
                reconstr_loss += tf.reduce_mean(min_error)
                if ssim_weight > 0:  # Also compute the minimum SSIM loss.
                    min_error_ssim = tf.minimum(ssim_error[s][key1],
                                              ssim_error[s][key2])
                    ssim_loss += tf.reduce_mean(min_error_ssim)

    # Build the total loss as composed of L1 reconstruction, SSIM, smoothing
    # and object size constraint loss as appropriate.
    reconstr_loss *= reconstr_weight
    total_loss = reconstr_loss
    if smooth_weight > 0:
        smooth_loss *= smooth_weight
        total_loss += smooth_loss
    if ssim_weight > 0:
        ssim_loss *= ssim_weight
        total_loss += ssim_loss

    return total_loss, warped_image, warp_mask