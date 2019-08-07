import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops


# Code modified from the following repos/resources:
#   * https://github.com/ppliuboy/SelFlow/blob/master/data_augmentation.py
#   * https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/#code
#   * https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
# All rights go to its rightful owner(s)
# TODO: Adapt to fix the Cuda-broken Data Augmentation
#   * see issues: https://github.com/sampepose/flownet2-tf/issues/14, https://github.com/sampepose/flownet2-tf/issues/30
# Changes to apply:
#   - Probably none or almost none (i.e.: test as is and fix any bugs that may occur)


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].
    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.
    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]


# We keep it as is as colour distortions can only be applied to img1 and optionally img2 (if we use image pairs)
def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
      image: 3-D Tensor containing single image in [0, 1].
      color_ordering: Python int, a type of distortion (valid values: 0-3).
      fast_mode: Avoids slower ops (random_hue and random_contrast)
      scope: Optional scope for name_scope.
    Returns:
      3-D Tensor color-distorted image on range [0, 1]
    Raises:
      ValueError: if color_ordering not in [0, 3]
    """
    max_delta_brightness = 51.  # around 0.2 specified in FlowNet2.0
    lower_saturation = 0.5  # FN2 samples from [0.5, 2] (halving/doubling saturation)
    upper_saturation = 1.5
    max_delta_hue = 0.2  # FN2 samples randomly for gamma instead
    lower_contrast = 0.2  # in flownet is [-0.8, 0.4] for a range of [-1, 1]?
    upper_contrast = 1.4
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=max_delta_brightness / 255.)
                image = tf.image.random_saturation(image, lower=lower_saturation, upper=upper_saturation)
            else:
                image = tf.image.random_saturation(image, lower=lower_saturation, upper=upper_saturation)
                image = tf.image.random_brightness(image, max_delta=max_delta_brightness / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=max_delta_brightness / 255.)
                image = tf.image.random_saturation(image, lower=lower_saturation, upper=upper_saturation)
                image = tf.image.random_hue(image, max_delta=max_delta_hue)
                image = tf.image.random_contrast(image, lower=lower_contrast, upper=upper_contrast)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=lower_saturation, upper=upper_saturation)
                image = tf.image.random_brightness(image, max_delta=max_delta_brightness / 255.)
                image = tf.image.random_contrast(image, lower=lower_contrast, upper=upper_contrast)
                image = tf.image.random_hue(image, max_delta=max_delta_hue)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=lower_contrast, upper=upper_contrast)
                image = tf.image.random_hue(image, max_delta=max_delta_hue)
                image = tf.image.random_brightness(image, max_delta=max_delta_brightness / 255.)
                image = tf.image.random_saturation(image, lower=lower_saturation, upper=upper_saturation)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=max_delta_hue)
                image = tf.image.random_saturation(image, lower=lower_saturation, upper=upper_saturation)
                image = tf.image.random_contrast(image, lower=lower_contrast, upper=upper_contrast)
                image = tf.image.random_brightness(image, max_delta=max_delta_brightness / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


# Note: not used ATM due to mismatch in net dimensions (first conv) in validation and training (tf.AUTO_REUSE)
def random_crop(img_list, crop_h, crop_w):
    img_size = tf.shape(img_list[0])
    # crop image and flow
    rand_offset_h = tf.random_uniform([], 0, img_size[0] - crop_h + 1, dtype=tf.int32)
    rand_offset_w = tf.random_uniform([], 0, img_size[1] - crop_w + 1, dtype=tf.int32)

    for i, img in enumerate(img_list):
        img_list[i] = tf.image.crop_to_bounding_box(img, rand_offset_h, rand_offset_w, crop_h, crop_w)

    return img_list


def flow_vertical_flip(flow):
    flow = tf.image.flip_up_down(flow)
    flow_u, flow_v = tf.unstack(flow, axis=-1)
    flow_v = flow_v * -1
    flow = tf.stack([flow_u, flow_v], axis=-1)
    return flow


def flow_horizontal_flip(flow):
    flow = tf.image.flip_left_right(flow)
    flow_u, flow_v = tf.unstack(flow, axis=-1)
    flow_u = flow_u * -1
    flow = tf.stack([flow_u, flow_v], axis=-1)
    return flow


def random_flip(img_list):
    is_flip = tf.random_uniform([2], minval=0, maxval=2, dtype=tf.int32)

    for i in range(len(img_list)):
        img_list[i] = tf.where(is_flip[0] > 0, tf.image.flip_left_right(img_list[i]), img_list[i])
        img_list[i] = tf.where(is_flip[1] > 0, tf.image.flip_up_down(img_list[i]), img_list[i])
    return img_list


def random_flip_with_flow(img_list, flow_list):
    is_flip = tf.random_uniform([2], minval=0, maxval=2, dtype=tf.int32)
    for i in range(len(img_list)):
        img_list[i] = tf.where(is_flip[0] > 0, tf.image.flip_left_right(img_list[i]), img_list[i])
        img_list[i] = tf.where(is_flip[1] > 0, tf.image.flip_up_down(img_list[i]), img_list[i])
    for i in range(len(flow_list)):
        flow_list[i] = tf.where(is_flip[0] > 0, flow_horizontal_flip(flow_list[i]), flow_list[i])
        flow_list[i] = tf.where(is_flip[1] > 0, flow_vertical_flip(flow_list[i]), flow_list[i])
    return img_list, flow_list


def random_channel_swap(img_list):
    channel_permutation = tf.constant([[0, 1, 2],
                                       [0, 2, 1],
                                       [1, 0, 2],
                                       [1, 2, 0],
                                       [2, 0, 1],
                                       [2, 1, 0]])
    rand_i = tf.random_uniform([], minval=0, maxval=6, dtype=tf.int32)
    permutation = channel_permutation[rand_i]

    # def channel_swap_once(image, perm):
    #     channel_1 = image[:, :, perm[0]]
    #     channel_2 = image[:, :, perm[1]]
    #     channel_3 = image[:, :, perm[2]]
    #     image = tf.stack([channel_1, channel_2, channel_3], axis=-1)
    #     return image
    #
    # img_list = tf.map_fn(lambda x: channel_swap_once(x, permutation), img_list)
    # for i, img in enumerate(img_list):
    for i in range(len(img_list)):
        img = img_list[i]
        channel_1 = img[:, :, permutation[0]]
        channel_2 = img[:, :, permutation[1]]
        channel_3 = img[:, :, permutation[2]]
        img_list[i] = tf.stack([channel_1, channel_2, channel_3], axis=-1)
    return img_list


def flow_resize(flow, out_size, is_scale=True, method=0):
    """
        method: 0 mean bilinear, 1 means nearest, 2 bicubic and 3 area
        See: https://www.tensorflow.org/api_docs/python/tf/image/ResizeMethod
    """
    flow_size = tf.to_float(tf.shape(flow)[-3:-1])
    flow = tf.image.resize_images(flow, out_size, method=method, align_corners=True)
    if is_scale:
        scale = tf.to_float(out_size) / flow_size
        scale = tf.stack([scale[1], scale[0]])
        flow = tf.multiply(flow, scale)
    return flow
