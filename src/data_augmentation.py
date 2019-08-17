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


def get_sampling_density(density, density_id=0, fast_mode=True, scope=None):
    with tf.name_scope(scope, 'get_density', [density]):
        if density_id == 0:
            density = tf.random_uniform([], minval=0.01, maxval=1., dtype=tf.float32)
        elif density_id == 1:
            density = tf.random_uniform([], minval=1., maxval=10., dtype=tf.float32)
        elif density_id == 2:
            density = tf.random_uniform([], minval=10., maxval=25., dtype=tf.float32)
        elif density_id == 3:
            density = tf.random_uniform([], minval=25., maxval=50., dtype=tf.float32)
        elif density_id == 4:
            density = tf.random_uniform([], minval=50., maxval=75., dtype=tf.float32)
        elif density_id == 5:
            density = tf.random_uniform([], minval=75., maxval=90., dtype=tf.float32)
        else:
            raise ValueError('density_id must be in [0, 5]')

        return density


def get_random_offset_and_crop(image_shape, density):
    """
    computes random crop sizes and offsets for a given image_shape (height, width) and sampling density
    :param image_shape:
    :param density:
    :return:
    """
    p_fill = density / 100  # target_density expressed in %
    bbox_area = p_fill * np.prod(image_shape)
    aspect_ratios = [16 / 9, 4 / 3, 3 / 2, 3 / 1, 4 / 5]
    aspect_id = np.random.choice(range(len(aspect_ratios)))
    aspect_ratio = aspect_ratios[aspect_id]
    # Compute width and height based of random aspect ratio and bbox area
    # bbox = w * h, AR = w/h
    crop_w = int(np.round(np.sqrt(bbox_area * aspect_ratio)))
    crop_h = int(np.round(crop_w / aspect_ratio))

    # Check crop dimensions are plausible, otherwise crop them to fit (this alters the density we were sampling at)
    if crop_h > image_shape[0] or crop_w > image_shape[1]:
        crop_h = image_shape[0] - 1 if crop_h > image_shape[0] else crop_h
        crop_w = image_shape[1] - 1 if crop_w > image_shape[1] else crop_w

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
    sparse_flow = tf.Variable(tf.zeros(gt_flow.shape, dtype=tf.float32), trainable=False)
    rand_offset_h, rand_offset_w, crop_h, crop_w = get_random_offset_and_crop((height, width), target_density)

    # Define matches as 0 inside the random bbox, 255s elsewhere (at training time the mask is normalised to [0,1])
    matches = 255 * np.ones((height, width), dtype=np.int32)  # (h, w)
    matches[rand_offset_h:rand_offset_h + crop_h, rand_offset_w: rand_offset_w + crop_w] = 0
    sampling_mask = matches
    matches = tf.cast(tf.expand_dims(matches, -1), dtype=tf.float32)  # convert to (h, w, 1)
    sampling_mask_rep = np.repeat(sampling_mask[:, :, np.newaxis], 2, axis=-1)
    sampling_mask_flatten = np.reshape(sampling_mask_rep, [-1])
    sampling_mask_flatten = np.where(sampling_mask_flatten == 255)

    gt_flow_sampling_mask = tf.boolean_mask(gt_flow, sampling_mask_rep)
    sparse_flow = tf.Variable(tf.reshape(sparse_flow, [-1]), trainable=False)
    sparse_flow = tf.scatter_update(sparse_flow, sampling_mask_flatten[0], gt_flow_sampling_mask)
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
    sparse_flow = tf.Variable(tf.zeros(gt_flow.shape, dtype=tf.float32), trainable=False)
    p_fill = target_density / 100  # target_density expressed in %
    sampling_mask = np.random.choice([0, 255], size=(height, width), p=[1 - p_fill, p_fill]).astype(
        np.int32)  # sampling_mask.shape: (h, w)
    matches = tf.cast(tf.expand_dims(255 * sampling_mask, -1), dtype=tf.float32)  # convert to (h, w, 1)
    sampling_mask_rep = np.repeat(sampling_mask[:, :, np.newaxis], 2, axis=-1)
    sampling_mask_flatten = np.reshape(sampling_mask_rep, [-1])
    sampling_mask_flatten = np.where(sampling_mask_flatten == 255)

    gt_flow_sampling_mask = tf.boolean_mask(gt_flow, sampling_mask_rep)
    sparse_flow = tf.Variable(tf.reshape(sparse_flow, [-1]), trainable=False)
    sparse_flow = tf.scatter_update(sparse_flow, sampling_mask_flatten[0], gt_flow_sampling_mask)
    sparse_flow = tf.reshape(sparse_flow, gt_flow.shape)

    return matches, sparse_flow


def sample_sparse_grid_like(gt_flow, target_density=75, height=384, width=512):
    """

    :param gt_flow:
    :param target_density:
    :return:
    """
    sparse_flow = tf.Variable(tf.zeros(gt_flow.shape, dtype=tf.float32), trainable=False)
    num_samples = (target_density / 100) * height * width
    aspect_ratio = width / height
    # Compute as in invalid_like for a random box to know the number of samples in horizontal and vertical
    num_samples_w = int(np.round(np.sqrt(num_samples * aspect_ratio)))
    num_samples_h = int(np.round(num_samples_w / aspect_ratio))

    # Check crop dimensions are plausible, otherwise crop them to fit (this alters the density we were sampling at)
    if num_samples_h > height or num_samples_w > width:
        num_samples_h = height if num_samples_h > height else num_samples_h
        num_samples_w = width if num_samples_w > width else num_samples_w

    sample_points_h = np.linspace(0, height - 1, num_samples_h, dtype=np.int32)
    sample_points_w = np.linspace(0, width - 1, num_samples_w, dtype=np.int32)
    # Create meshgrid of all combinations (i.e.: coordinates to sample at)
    matches = np.zeros((height, width), dtype=np.int32)
    xx, yy = np.meshgrid(sample_points_w, sample_points_h)
    xx_flatten = xx.flatten()
    yy_flatten = yy.flatten()
    matches[yy_flatten, xx_flatten] = 255

    # Randomly subtract a part with a random rectangle (superpixels in the future)
    if np.random.choice([0, 1]) == 0:
        rand_offset_h, rand_offset_w, crop_h, crop_w = get_random_offset_and_crop((height, width), target_density)
        matches[rand_offset_h:rand_offset_h + crop_h, rand_offset_w: rand_offset_w + crop_w] = 0
    sampling_mask = matches  # sampling_mask of size (h, w)
    matches = tf.cast(tf.expand_dims(matches, -1), dtype=tf.float32)  # convert to (h, w, 1)
    # Sample ground truth flow with given map
    sampling_mask_rep = np.repeat(sampling_mask[:, :, np.newaxis], 2, axis=-1)
    sampling_mask_flatten = np.reshape(sampling_mask_rep, [-1])
    sampling_mask_flatten = np.where(sampling_mask_flatten == 255)

    gt_flow_sampling_mask = tf.boolean_mask(gt_flow, sampling_mask_rep)
    sparse_flow = tf.Variable(tf.reshape(sparse_flow, [-1]), trainable=False)
    sparse_flow = tf.scatter_update(sparse_flow, sampling_mask_flatten[0], gt_flow_sampling_mask)
    sparse_flow = tf.reshape(sparse_flow, gt_flow.shape)

    return matches, sparse_flow


def sample_from_distribution(distrib_id, density, dm_matches, dm_flow, gt_flow):
    height, width, _ = gt_flow.get_shape().as_list()
    if distrib_id == 0:  # grid-like + random holes
        if density <= 1 and np.random.choice([0, 1]) == 0:  # use 'raw' DeepMatches
            sparse_flow = dm_flow
            matches = dm_matches
        else:
            matches, sparse_flow = sample_sparse_grid_like(gt_flow, target_density=density, height=height, width=width)

    elif distrib_id == 1:  # random uniform
        matches, sparse_flow = sample_sparse_uniform(gt_flow, target_density=density, height=height, width=width)

    elif distrib_id == 2:  # invalid-like (either 100% dense in known areas or 0% in unknown ones (holes))
        matches, sparse_flow = sample_sparse_invalid_like(gt_flow, target_density=density, height=height, width=width)
    else:
        raise ValueError("FATAL: id should have been an integer within (0-5) but instead was {}".format(distrib_id))

    return matches, sparse_flow


def sample_sparse_flow(dm_matches, dm_flow, gt_flow, num_ranges=6, num_distrib=3, fast_mode=False):
    """

    :param dm_matches:
    :param gt_flow:
    :return:
    """
    density = tf.zeros([], dtype=tf.float32)
    density = apply_with_random_selector(density, lambda x, ordering: get_sampling_density(x, ordering, fast_mode),
                                         num_cases=num_ranges)
    # # Sample a id which selects a "subrange" of density
    # density_id = np.random.choice(range(num_ranges))  # tf.random_uniform([], maxval=num_ranges, dtype=tf.int32)
    # if density_id == 0:  # very sparse matches (from 0.001 to 1% of the image area)
    #     density = np.random.uniform(low=0.01, high=1.)  # tf.random_uniform([], minval=0.01, maxval=1., dtype=tf.float32)
    # elif density_id == 1:  # quite sparse matches (from 1 to 10% of the image area)
    #     density = np.random.uniform(low=1., high=10.)  # tf.random_uniform([], minval=1., maxval=10., dtype=tf.float32)
    # elif density_id == 2:  # semi-sparse matches (from 10 to 25% of the image area)
    #     density = np.random.uniform(low=10., high=25.)  # tf.random_uniform([], minval=10., maxval=25., dtype=tf.float32)
    # elif density_id == 3:  # semi-dense matches (from 25 to 50% of the image area)
    #     density = np.random.uniform(low=25., high=50.)  # tf.random_uniform([], minval=25., maxval=50., dtype=tf.float32)
    # elif density_id == 4:  # quite dense matches (from 50 to 75% of the image area)
    #     density = np.random.uniform(low=50., high=75.)  # tf.random_uniform([], minval=50., maxval=75., dtype=tf.float32)
    # elif density_id == 5:  # very dense matches (from 75 to 90% of the image area)
    #     density = np.random.uniform(low=75., high=90.)  # tf.random_uniform([], minval=75., maxval=90., dtype=tf.float32)
    # else:
    #     raise ValueError("FATAL: id should have been an integer within (0-5) but instead was {}".format(density_id))
    # print("density_id: {}\ndensity: {:.2f}%".format(density_id, density))

    # Select a distribution (random uniform, invalid like or grid like with holes
    distrib_id = np.random.choice(range(num_distrib))  # tf.random_uniform([], maxval=num_distrib, dtype=tf.int32)
    print("distribution_id: {}".format(distrib_id))
    matches, sparse_flow = sample_from_distribution(distrib_id, density, dm_matches, dm_flow, gt_flow)

    return matches, sparse_flow
