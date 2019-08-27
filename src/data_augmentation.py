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


def distort_colour_zero(image, dist_params):
    image = tf.image.random_brightness(image, max_delta=dist_params['max_delta_brightness'] / 255.)
    image = tf.image.random_saturation(image, lower=dist_params['lower_saturation'],
                                       upper=dist_params['upper_saturation'])
    image = tf.image.random_hue(image, max_delta=dist_params['max_delta_hue'])
    image = tf.image.random_contrast(image, lower=dist_params['lower_contrast'], upper=dist_params['upper_contrast'])
    return image


def distort_colour_one(image, dist_params):
    image = tf.image.random_saturation(image, lower=dist_params['lower_saturation'],
                                       upper=dist_params['upper_saturation'])
    image = tf.image.random_brightness(image, max_delta=dist_params['max_delta_brightness'] / 255.)
    image = tf.image.random_contrast(image, lower=dist_params['lower_contrast'], upper=dist_params['upper_contrast'])
    image = tf.image.random_hue(image, max_delta=dist_params['max_delta_hue'])
    return image


def distort_colour_two(image, dist_params):
    image = tf.image.random_contrast(image, lower=dist_params['lower_contrast'], upper=dist_params['upper_contrast'])
    image = tf.image.random_hue(image, max_delta=dist_params['max_delta_hue'])
    image = tf.image.random_brightness(image, max_delta=dist_params['max_delta_brightness'] / 255.)
    image = tf.image.random_saturation(image, lower=dist_params['lower_saturation'],
                                       upper=dist_params['upper_saturation'])
    return image


def distort_colour_three(image, dist_params):
    image = tf.image.random_hue(image, max_delta=dist_params['max_delta_hue'])
    image = tf.image.random_saturation(image, lower=dist_params['lower_saturation'],
                                       upper=dist_params['upper_saturation'])
    image = tf.image.random_contrast(image, lower=dist_params['lower_contrast'], upper=dist_params['upper_contrast'])
    image = tf.image.random_brightness(image, max_delta=dist_params['max_delta_brightness'] / 255.)
    return image


# We keep it as is as colour distortions can only be applied to img1 and optionally img2 (if we use image pairs)
def distort_colour(image, num_permutations=4):
    distortion_params = {'max_delta_brightness': 51.,  # around 0.2 specified in FlowNet2.0
                         'lower_saturation': 0.5,  # FN2 samples from [0.5, 2] (halving/doubling saturation)
                         'upper_saturation': 1.5,
                         'max_delta_hue': 0.2,  # FN2 samples randomly for gamma instead
                         'lower_contrast': 0.2,  # in flownet is [-0.8, 0.4] for a range of [-1, 1]?
                         'upper_contrast': 1.4}

    colour_id = tf.random_uniform([], maxval=num_permutations, dtype=tf.int32)
    image = tf.case(
        {tf.equal(colour_id, tf.constant(0)): lambda: distort_colour_zero(image, dist_params=distortion_params),
         tf.equal(colour_id, tf.constant(1)): lambda: distort_colour_one(image, dist_params=distortion_params),
         tf.equal(colour_id, tf.constant(2)): lambda: distort_colour_two(image, dist_params=distortion_params),
         tf.equal(colour_id, tf.constant(3)): lambda: distort_colour_three(image, dist_params=distortion_params),
         },
        default=lambda: distort_colour_zero(image, dist_params=distortion_params), exclusive=True)
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

    for i in range(len(img_list)):
        img = img_list[i]
        channel_1 = img[:, :, permutation[0]]
        channel_2 = img[:, :, permutation[1]]
        channel_3 = img[:, :, permutation[2]]
        img_list[i] = tf.stack([channel_1, channel_2, channel_3], axis=-1)
    return img_list


def random_channel_swap_single(image):
    channel_permutation = tf.constant([[0, 1, 2],
                                       [0, 2, 1],
                                       [1, 0, 2],
                                       [1, 2, 0],
                                       [2, 0, 1],
                                       [2, 1, 0]])
    rand_i = tf.random_uniform([], minval=0, maxval=6, dtype=tf.int32)
    permutation = channel_permutation[rand_i]
    channel_1 = image[:, :, permutation[0]]
    channel_2 = image[:, :, permutation[1]]
    channel_3 = image[:, :, permutation[2]]
    image = tf.stack([channel_1, channel_2, channel_3], axis=-1)
    return image


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


def case_sparse(num_cases=2):
    density_id = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    density = tf.case(
        {tf.equal(density_id, tf.constant(0)): lambda: tf.random_uniform([], minval=0.01, maxval=1., dtype=tf.float32),
         tf.equal(density_id, tf.constant(1)): lambda: tf.random_uniform([], minval=1., maxval=10., dtype=tf.float32),
         },
        default=lambda: tf.random_uniform([], minval=1., maxval=10., dtype=tf.float32), exclusive=True)
    return density, density_id


def case_dense(num_cases=4):
    density_id = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    density = tf.case(
        {tf.equal(density_id, tf.constant(0)): lambda: tf.random_uniform([], minval=10., maxval=25., dtype=tf.float32),
         tf.equal(density_id, tf.constant(1)): lambda: tf.random_uniform([], minval=25., maxval=50., dtype=tf.float32),
         tf.equal(density_id, tf.constant(2)): lambda: tf.random_uniform([], minval=50., maxval=75., dtype=tf.float32),
         tf.equal(density_id, tf.constant(3)): lambda: tf.random_uniform([], minval=75., maxval=90., dtype=tf.float32),
         },
        default=lambda: tf.random_uniform([], minval=25., maxval=50., dtype=tf.float32), exclusive=True)
    return density, density_id


def get_sampling_density(dense_or_sparse, num_ranges=(4, 2)):
    density, density_id = tf.cond(tf.greater(dense_or_sparse, tf.constant(0)),
                                  lambda: case_dense(num_cases=num_ranges[0]),
                                  lambda: case_sparse(num_cases=num_ranges[1]))

    tf.summary.scalar('debug/density_id', density_id)
    return density


def get_random_offset_and_crop(image_shape, density):
    """
    computes random crop sizes and offsets for a given image_shape (height, width) and sampling density
    :param image_shape:
    :param density:
    :return:
    """
    p_fill = tf.divide(density, 100.0)  # target_density expressed in %
    bbox_area = tf.multiply(p_fill, tf.cast(tf.multiply(image_shape[0], image_shape[1]), dtype=tf.float32))
    num_aspect_ratios = 5
    aspect_ratios = tf.constant([16 / 9, 4 / 3, 3 / 2, 3 / 1, 4 / 5])
    aspect_id = tf.random_uniform([], maxval=num_aspect_ratios, dtype=tf.int32)
    aspect_ratio = aspect_ratios[aspect_id]
    # Compute width and height based of random aspect ratio and bbox area
    # bbox = w * h, AR = w/h

    # Check crop dimensions are plausible, otherwise crop them to fit (this alters the density we were sampling at)
    crop_w = tf.cast(tf.round(tf.sqrt(tf.multiply(tf.cast(bbox_area, dtype=tf.float32), aspect_ratio))), dtype=tf.int32)
    crop_h = tf.cast(tf.round(tf.divide(tf.cast(crop_w, dtype=tf.float32), aspect_ratio)), dtype=tf.int32)
    # crop_w = int(np.round(np.sqrt(bbox_area * aspect_ratio)))
    # crop_h = int(np.round(crop_w / aspect_ratio))

    # Check crop dimensions are plausible, otherwise crop them to fit (this alters the density we were sampling at)
    crop_h = tf.cond(tf.greater(crop_h, tf.constant(image_shape[0])),
                     lambda: tf.constant(image_shape[0] - 1), lambda: crop_h)
    crop_w = tf.cond(tf.greater(crop_w, tf.constant(image_shape[1])),
                     lambda: tf.constant(image_shape[1] - 1), lambda: crop_w)
    # if crop_h > image_shape[0] or crop_w > image_shape[1]:
    #     crop_h = image_shape[0] - 1 if crop_h > image_shape[0] else crop_h
    #     crop_w = image_shape[1] - 1 if crop_w > image_shape[1] else crop_w

    rand_offset_h = tf.random_uniform([], 0, image_shape[0] - crop_h + 1, dtype=tf.int32)
    rand_offset_w = tf.random_uniform([], 0, image_shape[1] - crop_w + 1, dtype=tf.int32)

    return rand_offset_h, rand_offset_w, crop_h, crop_w


# Functions to sample ground truth flow with different density and probability distribution
def sample_sparse_invalid_like(gt_flow, target_density=75, height=384, width=512):
    """

    :param gt_flow:
    :param target_density:
    :return:
    """
    # Important: matches is already normalised to [0, 1], only use those values
    # sparse_flow = tf.Variable(tf.zeros(gt_flow.shape, dtype=tf.float32), trainable=False)
    rand_offset_h, rand_offset_w, crop_h, crop_w = get_random_offset_and_crop((height, width), target_density)

    # Define matches as 0 inside the random bbox, 1s elsewhere
    ones = lambda: tf.ones((height * width), dtype=tf.float32)
    # matches = np.ones((height, width), dtype=np.int32)  # (h, w)
    # Assumption: matches is already a flatten array (when inputted to set_range...)
    matches = tf.Variable(initial_value=ones, dtype=tf.float32, trainable=False)
    # matches = tf.Variable(tf.reshape(matches, [-1]), trainable=False)
    matches = set_range_to_zero(matches, width, rand_offset_h, rand_offset_w, crop_h, crop_w)
    # Convert back to (height, width)
    matches = tf.reshape(matches, (height, width))
    # matches[rand_offset_h:rand_offset_h + crop_h, rand_offset_w: rand_offset_w + crop_w] = 0
    sampling_mask = matches
    matches = tf.cast(tf.expand_dims(matches, -1), dtype=tf.float32)  # convert to (h, w, 1)
    sampling_mask = sampling_mask[:, :, tf.newaxis]
    sampling_mask_rep = tf.tile(sampling_mask, [1, 1, 2])
    # sampling_mask_rep = np.repeat(sampling_mask[:, :, np.newaxis], 2, axis=-1)
    sampling_mask_flatten = tf.reshape(sampling_mask_rep, [-1])
    # sampling_mask_flatten = np.reshape(sampling_mask_rep, [-1])
    sampling_mask_flatten_where = tf.where(
        tf.equal(sampling_mask_flatten, tf.cast(1, dtype=sampling_mask_flatten.dtype)))
    sampling_mask_flatten_where = tf.reshape(sampling_mask_flatten_where, [-1])
    # sampling_mask_flatten = tf.where(tf.equal(sampling_mask_flatten, tf.cast(1, dtype=sampling_mask_flatten.dtype)))
    # sampling_mask_flatten = np.where(sampling_mask_flatten == 1)

    gt_flow_sampling_mask = tf.boolean_mask(gt_flow, sampling_mask_rep)
    zeros = lambda: tf.zeros(tf.reduce_prod(gt_flow.shape), dtype=tf.float32)
    sparse_flow = tf.Variable(initial_value=zeros, dtype=tf.float32, trainable=False)
    # sparse_flow = tf.Variable(tf.reshape(sparse_flow, [-1]), trainable=False)
    sparse_flow = tf.scatter_update(sparse_flow, sampling_mask_flatten_where, gt_flow_sampling_mask)
    sparse_flow = tf.reshape(sparse_flow, gt_flow.shape)

    return matches, sparse_flow


def sample_sparse_uniform(gt_flow, target_density=75, height=384, width=512):
    """

    Samples the provided gt flow with the target density and distribution. It also returns the updated matches mask
    which has 1s where the samples lay and 0 elsewhere.
    :param gt_flow: tensor containing ground truth optical flow (before batching ==> shape: (h, w, 2) )
    :param target_density:
    :return:
    """
    p_fill = tf.divide(target_density, 100.0)
    # p_fill = target_density / 100  # target_density expressed in %
    samples = tf.multinomial(tf.log([[1 - p_fill, p_fill]]), height * width)  # note log-prob
    sampling_mask = tf.cast(tf.reshape(samples, (height, width)), dtype=tf.int32)
    # sampling_mask = np.random.choice([0, 255], size=(height, width), p=[1 - p_fill, p_fill]).astype(
    #     np.int32)  # sampling_mask.shape: (h, w)
    matches = tf.cast(tf.expand_dims(sampling_mask, -1), dtype=tf.float32)  # convert to (h, w, 1)
    sampling_mask = sampling_mask[:, :, tf.newaxis]
    sampling_mask_rep = tf.tile(sampling_mask, [1, 1, 2])
    # sampling_mask_rep = np.repeat(sampling_mask[:, :, np.newaxis], 2, axis=-1)
    sampling_mask_flatten = tf.reshape(sampling_mask_rep, [-1])
    uniq = tf.unique(sampling_mask_flatten)
    # sampling_mask_flatten = np.reshape(sampling_mask_rep, [-1])
    sampling_mask_flatten_where = tf.where(tf.equal(sampling_mask_flatten, tf.cast(1, dtype=sampling_mask_flatten.dtype)))
    sampling_mask_flatten_where = tf.reshape(sampling_mask_flatten_where, [-1])

    # sampling_mask_flatten = np.where(sampling_mask_flatten == 1)
    gt_flow_sampling_mask = tf.boolean_mask(gt_flow, sampling_mask_rep)
    # gt_flow_boolean_mask = lambda: tf.boolean_mask(gt_flow, sampling_mask_rep)
    # gt_flow_sampling_mask = tf.Variable(initial_value=gt_flow_boolean_mask, dtype=tf.float32, validate_shape=False,
    #                                     trainable=False)
    zeros = lambda: tf.zeros(tf.reduce_prod(gt_flow.shape), dtype=tf.float32)
    sparse_flow = tf.Variable(initial_value=zeros, dtype=tf.float32, trainable=False)
    sparse_flow = tf.scatter_update(sparse_flow, sampling_mask_flatten_where, gt_flow_sampling_mask)
    sparse_flow = tf.reshape(sparse_flow, gt_flow.shape)

    return matches, sparse_flow


def set_range_to_zero(matches, width, offset_h, offset_w, crop_h, crop_w):
    range_rows = tf.range(offset_h, offset_h + crop_h, dtype=tf.int32)
    range_cols = tf.range(offset_w, offset_w + crop_w, dtype=tf.int32)
    rows, cols = tf.meshgrid(range_rows, range_cols)
    rows_flatten = tf.reshape(rows, [-1])
    cols_flatten = tf.reshape(cols, [-1])

    # Get absolute indices as rows * width + cols
    indices = tf.add(tf.multiply(rows_flatten, width), cols_flatten)
    zeros = tf.zeros(tf.shape(indices), dtype=tf.float32)
    matches = tf.scatter_update(matches, indices, zeros)
    # numpy: matches[rand_offset_h:rand_offset_h + crop_h, rand_offset_w: rand_offset_w + crop_w] = 0
    return matches


def corrupt_sparse_flow_loop(matches, density, height=384, width=512):
    def body(matches, density, height, width):  # what to do once the while loop condition is met
        matches = corrupt_sparse_flow_once(matches, density, height, width)
        return matches

    def condition(matches, density, height, width):
        return tf.greater(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.constant(0))

    # Perturbate always once (at least)
    matches = corrupt_sparse_flow_once(matches, density, height, width)
    # Draw a random number within 0, 1. If 1, keep corrupting the sparse flow (matches mask) with holes
    matches = tf.while_loop(condition, body, [matches, density, height, width])

    return matches


def corrupt_sparse_flow_once(matches, density, height=384, width=512):
    # Assumption: matches is already a flatten array
    inv_fraction = tf.random_uniform([], minval=4., maxval=12., dtype=tf.float32)
    rand_offset_h, rand_offset_w, crop_h, crop_w = get_random_offset_and_crop((height, width),
                                                                              tf.divide(density, inv_fraction))
    matches = set_range_to_zero(matches, width, rand_offset_h, rand_offset_w, crop_h, crop_w)
    return matches


def sample_sparse_grid_like(gt_flow, target_density=75, height=384, width=512):
    """

    :param gt_flow:
    :param target_density:
    :return:
    """
    # Important: matches is already normalised to [0, 1]
    # sparse_flow = tf.Variable(tf.zeros(gt_flow.shape, dtype=tf.float32), trainable=False)
    num_samples = tf.multiply(tf.multiply(tf.divide(target_density, 100.0), height), width)
    # num_samples = (target_density / 100) * height * width
    aspect_ratio = tf.divide(width, height)
    # Compute as in invalid_like for a random box to know the number of samples in horizontal and vertical
    num_samples_w = tf.cast(tf.round(tf.sqrt(tf.multiply(num_samples, aspect_ratio))),
                            dtype=tf.int32)
    # num_samples_w = int(np.round(np.sqrt(num_samples * aspect_ratio)))
    num_samples_h = tf.cast(tf.round(tf.divide(tf.cast(num_samples_w, dtype=tf.float32), aspect_ratio)),
                            dtype=tf.int32)
    # num_samples_h = int(np.round(num_samples_w / aspect_ratio))

    # Check crop dimensions are plausible, otherwise crop them to fit (this alters the density we were sampling at)
    num_samples_h = tf.cond(
        tf.greater(num_samples_h, tf.constant(height)), lambda: tf.constant(height, dtype=tf.int32),
        lambda: num_samples_h)
    num_samples_w = tf.cond(
        tf.greater(num_samples_w, tf.constant(width)), lambda: tf.constant(width, dtype=tf.int32),
        lambda: num_samples_w)
    # if num_samples_h > height or num_samples_w > width:
    #     num_samples_h = height if num_samples_h > height else num_samples_h
    #     num_samples_w = width if num_samples_w > width else num_samples_w
    delta_rows = tf.cast((height - 1 - 0) / num_samples_h, tf.float32)
    sample_points_h = tf.cast(tf.round(tf.range(start=0, limit=height, delta=delta_rows, dtype=tf.float32)),
                              dtype=tf.int32)
    delta_cols = tf.cast((width - 1 - 0) / num_samples_w, tf.float32)
    sample_points_w= tf.cast(tf.round(tf.range(start=0, limit=width, delta=delta_cols, dtype=tf.float32)),
                             dtype=tf.int32)
    # Create meshgrid of all combinations (i.e.: coordinates to sample at)
    rows, cols = tf.meshgrid(sample_points_h, sample_points_w, indexing='ij')
    # xx, yy = np.meshgrid(sample_points_w, sample_points_h)
    rows_flatten = tf.reshape(rows, [-1])
    # xx_flatten = xx.flatten()
    cols_flatten = tf.reshape(cols, [-1])
    # yy_flatten = yy.flatten()

    # Compute absolute indices as row * width + cols
    indices = tf.add(tf.multiply(rows_flatten, width), cols_flatten)
    # ones_raw = lambda: tf.ones(tf.shape(indices))
    # ones = tf.Variable(initial_value=ones_raw, trainable=False, validate_shape=False)
    ones = tf.ones(tf.shape(indices), dtype=tf.float32)
    zeros = lambda: tf.zeros((height * width), dtype=tf.float32)
    matches = tf.Variable(initial_value=zeros, trainable=False)
    # matches = np.zeros((height, width), dtype=np.int32)

    matches = tf.scatter_update(matches, indices, ones)  # all 1D tensors
    # matches[yy_flatten, xx_flatten] = 1
    # matches = tf.reshape(matches, (height, width, 1))

    # Randomly subtract a part with a random rectangle (superpixels in the future)
    corrupt_mask = tf.random_uniform([], maxval=2, dtype=tf.int32)  # np.random.choice([0, 1])
    matches = tf.cond(tf.greater(corrupt_mask, tf.constant(0)), lambda: corrupt_sparse_flow_once(matches, target_density,
                                                                                                 height, width),
                      lambda: return_identity_one(matches))

    sampling_mask = tf.reshape(matches, (height, width))  # sampling_mask of size (h, w)
    matches = tf.cast(tf.expand_dims(sampling_mask, -1), dtype=tf.float32)  # convert to (h, w, 1)
    # Sample ground truth flow with given map
    sampling_mask = sampling_mask[:, :, tf.newaxis]
    sampling_mask_rep = tf.tile(sampling_mask, [1, 1, 2])
    # sampling_mask_rep = np.repeat(sampling_mask[:, :, np.newaxis], 2, axis=-1)
    sampling_mask_flatten = tf.reshape(sampling_mask_rep, [-1])
    # sampling_mask_flatten = np.reshape(sampling_mask_rep, [-1])
    sampling_mask_flatten_where = tf.where(
        tf.equal(sampling_mask_flatten, tf.cast(1, dtype=sampling_mask_flatten.dtype)))
    sampling_mask_flatten_where = tf.reshape(sampling_mask_flatten_where, [-1])
    # sampling_mask_flatten = tf.where(tf.equal(sampling_mask_flatten, tf.cast(1, dtype=sampling_mask_flatten.dtype)))
    # sampling_mask_flatten = np.where(sampling_mask_flatten == 1)

    gt_flow_sampling_mask = tf.boolean_mask(gt_flow, sampling_mask_rep)
    zeros = lambda: tf.zeros(tf.reduce_prod(gt_flow.shape), dtype=tf.float32)
    sparse_flow = tf.Variable(initial_value=zeros, dtype=tf.float32, trainable=False)
    # sparse_flow = tf.Variable(tf.reshape(sparse_flow, [-1]), trainable=False)
    sparse_flow = tf.scatter_update(sparse_flow, sampling_mask_flatten_where, gt_flow_sampling_mask)
    sparse_flow = tf.reshape(sparse_flow, gt_flow.shape)

    return matches, sparse_flow


def return_identity(x, y):
    return tf.identity(x), tf.identity(y)


def return_identity_one(x):
    return tf.identity(x)


def sample_from_distribution(distrib_id, density, dm_matches, dm_flow, gt_flow):
    default_density = 25  # default density to use with default uniform sampling
    height, width, _ = gt_flow.get_shape().as_list()
    aux_choice = tf.random_uniform([], maxval=2, dtype=tf.int32)  # 0 or 1
    sample_dm = tf.cond(tf.logical_and(tf.greater(aux_choice, tf.constant(0)),
                        tf.less_equal(density, tf.constant(1.0))), lambda: tf.constant(True), lambda: tf.constant(False))
    # sample_dm = tf.cond(True if (np.random.choice([0, 1]) > 0 and density <= 1) else False
    matches, sparse_flow = tf.case(
        {
            tf.logical_and(tf.equal(distrib_id, tf.constant(0)),
                           tf.equal(sample_dm, tf.constant(False))): lambda: sample_sparse_grid_like(
                gt_flow, target_density=density, height=height, width=width),
            tf.logical_and(tf.equal(distrib_id, tf.constant(0)),
                           tf.equal(sample_dm, tf.constant(True))): lambda: return_identity(dm_matches, dm_flow),
            tf.equal(distrib_id, tf.constant(1)): lambda: sample_sparse_uniform(gt_flow, target_density=density,
                                                                                height=height, width=width),
            tf.equal(distrib_id, tf.constant(2)): lambda: sample_sparse_invalid_like(gt_flow, target_density=density,
                                                                                     height=height, width=width)
        },
        default=lambda: sample_sparse_uniform(gt_flow, target_density=default_density, height=height, width=width),
        exclusive=True)

    # Ensure we do not give an empty mask back!
    matches, sparse_flow = tf.cond(tf.greater(tf.reduce_sum(matches), 0.0),
                                   lambda: return_identity(matches, sparse_flow),
                                   lambda: return_identity(dm_matches, dm_flow))

    return matches, sparse_flow


def sample_sparse_flow(dm_matches, dm_flow, gt_flow, num_ranges=(4, 2), num_distrib=3, fast_mode=False):
    """

    :param dm_matches:
    :param gt_flow:
    :return:
    """
    # apply_with_random_selector does not work for our case (maybe it interferes with tf.slim or something else...)
    # use tf.case instead
    # density = apply_with_random_selector(density, lambda x, ordering: get_sampling_density(x, ordering, fast_mode),
    #                                      num_cases=num_ranges)
    dense_or_sparse = tf.random_uniform([], maxval=2, dtype=tf.int32)  # 0 or 1
    density = get_sampling_density(dense_or_sparse, num_ranges=num_ranges)
    tf.summary.scalar('debug/density', density)

    # Select a distribution (random uniform, invalid like or grid like with holes
    distrib_id = tf.random_uniform([], maxval=num_distrib, dtype=tf.int32)  # np.random.choice(range(num_distrib))
    tf.summary.scalar('debug/distrib_id', distrib_id)
    matches, sparse_flow = sample_from_distribution(distrib_id, density, dm_matches, dm_flow, gt_flow)

    return matches, sparse_flow
