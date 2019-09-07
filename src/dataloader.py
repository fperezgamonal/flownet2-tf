# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import copy
slim = tf.contrib.slim
from math import exp
from .dataset_configs import FLYING_CHAIRS_ALL_DATASET_CONFIG, SINTEL_FINAL_ALL_DATASET_CONFIG,\
    SINTEL_ALL_DATASET_CONFIG, FLYING_THINGS_3D_ALL_DATASET_CONFIG, FC_TRAIN_SINTEL_VAL_DATASET_CONFIG,\
    FT3D_TRAIN_SINTEL_VAL_DATASET_CONFIG, FLYING_CHAIRS_MINI_DATASET_CONFIG, FT3D_TRAIN_SINTEL_VAL_MINI_DATASET_CONFIG,\
    FC_TRAIN_SINTEL_VAL_MINI_DATASET_CONFIG, FLYING_THINGS_3D_MINI_DATASET_CONFIG, SINTEL_MINI_DATASET_CONFIG, \
    ALLEY_MINI_DATASET_CONFIG
from src.data_augmentation import *
from src.flowlib import flow_to_image

_preprocessing_ops = tf.load_op_library(
    tf.resource_loader.get_path_to_datafile("./ops/build/preprocessing.so"))

# Mean DeepMatching density for training/validation sets (in percentage % of the num_matches/image_area)
fc_sparse_perc = 0.267129
ft3d_sparse_perc = 0.249715
sintel_sparse_perc = 0.222320


# Code modified from the following repos:
#   * https://github.com/ppliuboy/DDFlow/blob/master/datasets.py
def augment_image_pair(img1, img2, crop_h, crop_w):
    img1, img2 = random_crop([img1, img2], crop_h, crop_w)
    img1, img2 = random_flip([img1, img2])
    img1, img2 = random_channel_swap([img1, img2])
    return img1, img2


# add_summary shows images when they are being augmented (that is, several images from the same batch)
# This might be too verbose once we known the augmentations work as expected since the train/image_a, etc. are augmented
# already and shown in TB by default
def augment_all_interp(image, matches, sparse_flow, edges, gt_flow, crop_h, crop_w, add_summary=False, fast_mode=False,
                       global_step=None, invalid_like=-1, num_distrib=2, fixed_density=-1.0):
    # print_out = tf.cond(tf.equal(global_step, tf.cast(tf.constant(0), tf.int64)),
    #                     lambda: tf.print("global_step: {}".format(global_step)), lambda: tf.print("Not 0"))
    # tf.print("global_step: {}".format(global_step))
    # Check if we can get global step value without explicitly passing it in which broke restoration from checkpoint
    # print("global_step.eval()".format(tf.train.get_global_step(graph=None).eval()))
    # num_distributions = 2  # only uniform or deep matching for now (as mixing invalid-like may be too different and
    # grid-like did not work as expected

    # If num_distrib = -1, use uniform sampling instead with fixed_density which must be > 0
    # fixed_density_tensor = tf.convert_to_tensor(fixed_density)
    # height, width, _ = gt_flow.get_shape().as_list()
    # matches, sparse_flow = tf.cond(tf.logical_and(tf.equal(tf.convert_to_tensor(num_distrib), tf.constant(-1)),
    #                                               tf.greater(fixed_density_tensor, tf.constant(0.0))),
    #                                lambda: sample_sparse_uniform(gt_flow=gt_flow, target_density=fixed_density_tensor,
    #                                                              height=height, width=width),
    #                                lambda: sample_sparse_flow(matches, sparse_flow, gt_flow, invalid_like=invalid_like,
    #                                                           num_distrib=num_distrib))
    matches, sparse_flow = sample_sparse_flow(matches, sparse_flow, gt_flow, invalid_like=invalid_like,
                                              num_distrib=num_distrib, fixed_density=fixed_density)
    # image, matches, sparse_flow, edges, gt_flow = random_crop([image, matches, sparse_flow, edges, gt_flow], crop_h,
    #                                                           crop_w)
    # Random flip of images and flow
    [image, matches, edges], [sparse_flow, gt_flow] = random_flip_with_flow([image, matches, edges], [sparse_flow,
                                                                                                      gt_flow])
    # Random channel swap (only for RGB images)
    image = random_channel_swap_single(image)

    # Random colour distortions (only for RGB images)
    # There are 1 or 4 ways to do it. (1 == only brightness and contrast, 4 hue and saturation)
    num_distort_cases = 1 if fast_mode else 4
    image = distort_colour(image, num_permutations=num_distort_cases)
    if add_summary:
        tf.summary.image('data_augmentation/distorted_image',
                         tf.expand_dims(image, 0))
        tf.summary.image('data_augmentation/distorted_matches',
                         tf.expand_dims(matches, 0))
        tf.summary.image('data_augmentation/distorted_edges',
                         tf.expand_dims(edges, 0))
        # Flow to RGB representation
        gt_flow_img = tf.py_func(flow_to_image, [gt_flow], tf.uint8)
        tf.summary.image('data_augmentation/true_flow', tf.expand_dims(gt_flow_img, 0), max_outputs=1)

    return image, matches, sparse_flow, edges, gt_flow


def augment_all_estimation(image1, image2, gt_flow, crop_h, crop_w, add_summary=False, fast_mode=False):
    # image, image2, gt_flow = random_crop([image1, image2, gt_flow], crop_h, crop_w)
    [image1, image2], gt_flow = random_flip_with_flow([image1, image2], gt_flow)
    image1, image2 = random_channel_swap([image1, image2])  # only for 'image-like' inputs (not flow)

    # Random colour distortions (only for RGB images)
    # There are 1 or 4 ways to do it. (1 == only brightness and contrast, 4 hue and saturation)
    num_distort_cases = 1 if fast_mode else 4
    image1 = distort_colour(image1, num_permutations=num_distort_cases)
    image2 = distort_colour(image2, num_permutations=num_distort_cases)
    if add_summary:
        tf.summary.image('data_augmentation/distorted_image1',
                         tf.expand_dims(image1, 0))
        tf.summary.image('data_augmentation/distorted_image2',
                         tf.expand_dims(image2, 0))
        # Flow to RGB representation
        true_flow_0 = gt_flow[0, :, :, :]
        true_flow_img = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)

        tf.summary.image('data_augmentation/true_flow', tf.expand_dims(true_flow_img, 0), max_outputs=1)
    return image1, image2, gt_flow


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/data/tfexample_decoder.py
class Image(slim.tfexample_decoder.ItemHandler):
    """An ItemHandler that decodes a parsed Tensor as an image."""

    def __init__(self,
                 image_key=None,
                 format_key=None,
                 shape=None,
                 channels=3,
                 dtype=tf.uint8,
                 repeated=False):
        """Initializes the image.
        Args:
          image_key: the name of the TF-Example feature in which the encoded image
            is stored.
          shape: the output shape of the image as 1-D `Tensor`
            [height, width, channels]. If provided, the image is reshaped
            accordingly. If left as None, no reshaping is done. A shape should
            be supplied only if all the stored images have the same shape.
          channels: the number of channels in the image.
          dtype: images will be decoded at this bit depth. Different formats
            support different bit depths.
              See tf.image.decode_image,
                  tf.decode_raw,
          repeated: if False, decodes a single image. If True, decodes a
            variable number of image strings from a 1D tensor of strings.
        """
        if not image_key:
            image_key = 'image/encoded'

        super(Image, self).__init__([image_key])
        self._image_key = image_key
        self._shape = shape
        self._channels = channels
        self._dtype = dtype
        self._repeated = repeated

    def tensors_to_item(self, keys_to_tensors):
        """See base class."""
        image_buffer = keys_to_tensors[self._image_key]

        if self._repeated:
            return functional_ops.map_fn(lambda x: self._decode(x),
                                         image_buffer, dtype=self._dtype)
        else:
            return self._decode(image_buffer)

    def _decode(self, image_buffer):
        """Decodes the image buffer.
        Args:
          image_buffer: The tensor representing the encoded image tensor.
        Returns:
          A tensor that represents decoded image of self._shape, or
          (?, ?, self._channels) if self._shape is not specified.
        """
        def decode_raw():
            """Decodes a raw image."""
            return tf.decode_raw(image_buffer, out_type=self._dtype)

        image = decode_raw()
        # image.set_shape([None, None, self._channels])
        if self._shape is not None:
            image = tf.reshape(image, self._shape)

        return image


def __get_dataset(dataset_config, split_name, input_type='image_pairs'):
    """
    dataset_config: A dataset_config defined in dataset_configs.py
    split_name: 'train'/'valid'
    """
    with tf.name_scope('__get_dataset'):
        if split_name not in dataset_config['SIZES']:
            raise ValueError('split name {} not recognized'.format(split_name))

        # Width and height accounting for needed padding to match network dimensions
        # Different origins for train and val
        if dataset_config == FC_TRAIN_SINTEL_VAL_DATASET_CONFIG or \
                dataset_config == FT3D_TRAIN_SINTEL_VAL_DATASET_CONFIG or \
                dataset_config == FC_TRAIN_SINTEL_VAL_MINI_DATASET_CONFIG or \
                dataset_config == FT3D_TRAIN_SINTEL_VAL_MINI_DATASET_CONFIG:
            print("Dataset selected uses two different origins for train and validation images(make sure it is OK!)")
            if split_name == 'train':
                image_height, image_width = dataset_config['PADDED_IMAGE_HEIGHT'][0], \
                                            dataset_config['PADDED_IMAGE_WIDTH'][0]
            elif split_name == 'valid':
                image_height, image_width = dataset_config['PADDED_IMAGE_HEIGHT'][1], \
                                            dataset_config['PADDED_IMAGE_WIDTH'][1]
            else:
                raise ValueError("FATAL: unexpected 'split_name'. Must be either 'train' or 'valid'")

        else:  # train and val are subsets of the same set
            image_height, image_width = dataset_config['PADDED_IMAGE_HEIGHT'], dataset_config['PADDED_IMAGE_WIDTH']
        reader = tf.TFRecordReader
        if input_type == 'image_matches':
            keys_to_features = {
                'image_a': tf.FixedLenFeature([], tf.string),
                'matches_a': tf.FixedLenFeature([], tf.string),
                'sparse_flow': tf.FixedLenFeature([], tf.string),
                'edges_a': tf.FixedLenFeature([], tf.string),
                'flow': tf.FixedLenFeature([], tf.string),
            }
            items_to_handlers = {
                'image_a': Image(
                    image_key='image_a',
                    dtype=tf.float64,
                    shape=[image_height, image_width, 3],
                    channels=3),
                'matches_a': Image(
                    image_key='matches_a',
                    dtype=tf.float64,
                    shape=[image_height, image_width, 1],
                    channels=1),
                'sparse_flow': Image(
                    image_key='sparse_flow',
                    dtype=tf.float32,
                    shape=[image_height, image_width, 2],
                    channels=2),
                'edges_a': Image(
                    image_key='edges_a',
                    dtype=tf.float32,
                    shape=[image_height, image_width, 1],
                    channels=1),
                'flow': Image(
                    image_key='flow',
                    dtype=tf.float32,
                    shape=[image_height, image_width, 2],
                    channels=2),
            }
        else:
            keys_to_features = {
                'image_a': tf.FixedLenFeature([], tf.string),
                'image_b': tf.FixedLenFeature([], tf.string),
                'flow': tf.FixedLenFeature([], tf.string),
            }
            items_to_handlers = {
                'image_a': Image(
                    image_key='image_a',
                    dtype=tf.float64,
                    shape=[image_height, image_width, 3],
                    channels=3),
                'image_b': Image(
                    image_key='image_b',
                    dtype=tf.float64,
                    shape=[image_height, image_width, 3],
                    channels=3),
                'flow': Image(
                    image_key='flow',
                    dtype=tf.float32,
                    shape=[image_height, image_width, 2],
                    channels=2),
            }

        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
        return slim.dataset.Dataset(
            data_sources=dataset_config['PATHS'][split_name],
            reader=reader,
            decoder=decoder,
            num_samples=dataset_config['SIZES'][split_name],
            items_to_descriptions=dataset_config['ITEMS_TO_DESCRIPTIONS'])


def config_to_arrays(dataset_config):
    output = {
        'name': [],
        'rand_type': [],
        'exp': [],
        'mean': [],
        'spread': [],
        'prob': [],
        'coeff_schedule': [],
    }
    config = copy.deepcopy(dataset_config)

    if 'coeff_schedule_param' in config:
        del config['coeff_schedule_param']

    # Get all attributes
    for (name, value) in config.items():
        if name == 'coeff_schedule_param':
            output['coeff_schedule'] = [value['half_life'],
                                        value['initial_coeff'],
                                        value['final_coeff']]
        else:
            output['name'].append(name)
            output['rand_type'].append(value['rand_type'])
            output['exp'].append(value['exp'])
            output['mean'].append(value['mean'])
            output['spread'].append(value['spread'])
            output['prob'].append(value['prob'])

    return output


# https://github.com/tgebru/transform/blob/master/src/caffe/layers/data_augmentation_layer.cpp#L34
def _generate_coeff(param, discount_coeff=tf.constant(1.0), default_value=tf.constant(0.0)):
    if not all(name in param for name in ['rand_type', 'exp', 'mean', 'spread', 'prob']):
        raise RuntimeError('Expected rand_type, exp, mean, spread, prob in `param`')

    rand_type = param['rand_type']
    exp = float(param['exp'])
    mean = tf.convert_to_tensor(param['mean'], dtype=tf.float32)
    spread = float(param['spread'])  # AKA standard deviation
    prob = float(param['prob'])

    # Multiply spread by our discount_coeff so it changes over time
    spread = spread * discount_coeff

    if rand_type == 'uniform':
        value = tf.cond(spread > 0.0,
                        lambda: tf.random_uniform([], mean - spread, mean + spread),
                        lambda: mean)
        if exp:
            value = tf.exp(value)
    elif rand_type == 'gaussian':
        value = tf.cond(spread > 0.0,
                        lambda: tf.random_normal([], mean, spread),
                        lambda: mean)
        if exp:
            value = tf.exp(value)
    elif rand_type == 'bernoulli':
        if prob > 0.0:
            value = tf.contrib.distributions.Bernoulli(probs=prob).sample([])
        else:
            value = 0.0
    elif rand_type == 'uniform_bernoulli':
        tmp1 = 0.0
        tmp2 = 0
        if prob > 0.0:
            tmp2 = tf.contrib.distributions.Bernoulli(probs=prob).sample([])
        else:
            tmp2 = 0

        if tmp2 == 0:
            if default_value is not None:
                return default_value
        else:
            tmp1 = tf.cond(spread > 0.0,
                           lambda: tf.random_uniform([], mean - spread, mean + spread),
                           lambda: mean)
        if exp:
            tmp1 = tf.exp(tmp1)
        value = tmp1
    elif rand_type == 'gaussian_bernoulli':
        tmp1 = 0.0
        tmp2 = 0
        if prob > 0.0:
            tmp2 = tf.contrib.distributions.Bernoulli(probs=prob).sample([])
        else:
            tmp2 = 0

        if tmp2 == 0:
            if default_value is not None:
                return default_value
        else:
            tmp1 = tf.cond(spread > 0.0,
                           lambda: tf.random_normal([], mean, spread),
                           lambda: mean)
        if exp:
            tmp1 = tf.exp(tmp1)
        value = tmp1
    else:
        raise ValueError('Unknown distribution type {}.'.format(rand_type))
    return value


# TODO: add possibility of using uniform sampling as default matches w/o mixing (did not work with commented changes)
def load_batch(dataset_config_str, split_name, global_step=None, input_type='image_pairs', common_queue_capacity=64,
               common_queue_min=32, capacity_in_batches_train=4, capacity_in_batches_val=1, num_threads=8,
               batch_size=None, data_augmentation=True, add_summary_augmentation=False, invalid_like=-1,
               num_distrib=2):

    num_distrib_tensor = tf.convert_to_tensor(num_distrib)
    if dataset_config_str.lower() == 'flying_things3d':
        dataset_config = FLYING_THINGS_3D_ALL_DATASET_CONFIG
        train_dataset = 'ft3d'
        val_dataset = 'ft3d'
    elif dataset_config_str.lower() == 'sintel_final':
        dataset_config = SINTEL_FINAL_ALL_DATASET_CONFIG
        train_dataset = 'sintel'
        val_dataset = 'sintel'
    elif dataset_config_str.lower() == 'sintel_all':  # clean + final
        dataset_config = SINTEL_ALL_DATASET_CONFIG
        train_dataset = 'sintel'
        val_dataset = 'sintel'
    elif dataset_config_str.lower() == 'fc_sintel':  # FC (train) + Sintel (validation)
        dataset_config = FC_TRAIN_SINTEL_VAL_DATASET_CONFIG
        train_dataset = 'fc'
        val_dataset = 'sintel'
    elif dataset_config_str.lower() == 'ft3d_sintel':  # FT3D (train) + Sintel (validation)
        dataset_config = FT3D_TRAIN_SINTEL_VAL_DATASET_CONFIG
        train_dataset = 'ft3d'
        val_dataset = 'sintel'
    elif dataset_config_str.lower() == 'fc_mini':
        dataset_config = FLYING_CHAIRS_MINI_DATASET_CONFIG
        train_dataset = 'fc'
        val_dataset = 'fc'
    elif dataset_config_str.lower() == 'ft3d_mini':
        dataset_config = FLYING_THINGS_3D_MINI_DATASET_CONFIG
        train_dataset = 'ft3d'
        val_dataset = 'ft3d'
    elif dataset_config_str.lower() == 'sintel_mini':
        dataset_config = SINTEL_MINI_DATASET_CONFIG
        train_dataset = 'sintel'
        val_dataset = 'sintel'
    elif dataset_config_str.lower() == 'fc_sintel_mini':
        dataset_config = FC_TRAIN_SINTEL_VAL_MINI_DATASET_CONFIG
        train_dataset = 'fc'
        val_dataset = 'sintel'
    elif dataset_config_str.lower() == 'ft3d_sintel_mini':
        dataset_config = FT3D_TRAIN_SINTEL_VAL_MINI_DATASET_CONFIG
        train_dataset = 'ft3d'
        val_dataset = 'sintel'
    else:  # flying_chairs
        dataset_config = FLYING_CHAIRS_ALL_DATASET_CONFIG
        train_dataset = 'fc'
        val_dataset = 'fc'

    if batch_size is not None and batch_size > 0:
        print("Batch size changed from training_schedules.py default '{}' to '{}'".format(
            dataset_config['BATCH_SIZE'], batch_size))
    else:
        print("Batch size kept to dataset default value: {}".format(dataset_config['BATCH_SIZE']))
        batch_size = dataset_config['BATCH_SIZE']

    reader_kwargs = {'options': tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)}
    with tf.name_scope('load_batch'):
        dataset = __get_dataset(dataset_config, split_name, input_type=input_type)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=num_threads,
            common_queue_capacity=common_queue_capacity,  # this also broke training, we lowered it (og. value = 2048)
            common_queue_min=common_queue_min,  # this also broke training, we lowered it (og. value = 1024)
            reader_kwargs=reader_kwargs,
            shuffle=True if split_name == 'train' else False,)

        if input_type == 'image_matches':
            image_b = None
            image_a_og, matches_a_og, sparse_flow_og, edges_a_og, gt_flow_og = data_provider.get(['image_a', 'matches_a',
                                                                                                  'sparse_flow',
                                                                                                  'edges_a', 'flow'])

            image_a_og, matches_a_og, edges_a_og = map(lambda x: tf.cast(x, dtype=tf.float32), [image_a_og, matches_a_og,
                                                                                                edges_a_og])
        else:
            matches_a_og = None
            sparse_flow_og = None
            edges_a_og = None
            image_a_og, image_b_og, gt_flow_og = data_provider.get(['image_a', 'image_b', 'flow'])
            image_a_og, image_b_og = map(lambda x: tf.cast(x, dtype=tf.float32), [image_a_og, image_b_og])

        if dataset_config['PREPROCESS']['scale']:  # if record data has not been scaled, 'scale' should be True
            image_a_og = image_a_og / 255.0
            if input_type == 'image_matches':  # Note: we assume edges_a to be in the range [0,1] in both cases
                matches_a_og = matches_a_og / 255.0
            else:
                image_b_og = image_b_og / 255.0

        if data_augmentation and split_name == 'train':
            crop = [dataset_config['PREPROCESS']['crop_height'], dataset_config['PREPROCESS']['crop_width']]

        # Data Augmentation (SelFlow functions work on a 'standard' 3D-array (still not batched) => (h, w, ch)
        if input_type == 'image_matches':
            if data_augmentation and split_name == 'train':
                if train_dataset.lower() == 'sintel':
                    target_density = sintel_sparse_perc
                elif train_dataset.lower() == 'ft3d':
                    target_density = ft3d_sparse_perc
                else:  # FC
                    target_density = fc_sparse_perc

                print("(image_matches) Applying data augmentation...")  # temporally to debug
                image_a, matches_a, sparse_flow, edges_a, gt_flow = augment_all_interp(
                    image_a_og, matches_a_og, sparse_flow_og, edges_a_og, gt_flow_og, crop_h=crop[0], crop_w=crop[1],
                    add_summary=add_summary_augmentation, fast_mode=False, global_step=global_step,
                    invalid_like=invalid_like, num_distrib=num_distrib, fixed_density=target_density)
            else:
                if val_dataset.lower() == 'fc':
                    target_density = fc_sparse_perc
                elif val_dataset.lower() == 'ft3d':
                    target_density = ft3d_sparse_perc
                else:  # Sintel
                    target_density = sintel_sparse_perc

                # If num_distrib=-1, sample uniformly instead of using DM matches
                # height, width, _ = flow.get_shape().as_list()
                # matches_a, sparse_flow = tf.cond(tf.equal(num_distrib_tensor, tf.constant(-1)),
                #                                  lambda: sample_sparse_uniform(gt_flow=flow,
                #                                                                target_density=target_density,
                #                                                                height=height, width=width),
                #                                  lambda: return_identity(matches_a, sparse_flow))
            # Add extra 'batching' dimension
            image_bs = None
            image_as, matches_as, sparse_flows, edges_as, flows = map(lambda x: tf.expand_dims(x, 0),
                                                                      [image_a, matches_a, sparse_flow, edges_a,
                                                                       gt_flow])

        else:
            if data_augmentation and split_name == 'train':
                print("(image_pairs) Applying data augmentation...")  # temporally to debug
                image_a, image_b, gt_flow = augment_all_estimation(image_a_og, image_b_og, gt_flow_og, crop_h=crop[0],
                                                                   crop_w=crop[1], add_summary=add_summary_augmentation,
                                                                   fast_mode=False)
            matches_as = None
            sparse_flows = None
            edges_as = None
            image_as, image_bs, flows = map(lambda x: tf.expand_dims(x, 0), [image_a, image_b, gt_flow])


        if input_type == 'image_matches':
            print("(dataloader.py): input_type is 'image_matches'; split is '{}'".format(split_name))
            if split_name == 'valid':
                return tf.train.batch([image_as, matches_as, sparse_flows, edges_as, flows],
                                      enqueue_many=True,
                                      batch_size=batch_size,
                                      capacity=batch_size * capacity_in_batches_val,
                                      allow_smaller_final_batch=False,
                                      num_threads=num_threads)
            else:

                return tf.train.batch([image_as, matches_as, sparse_flows, edges_as, flows],
                                      enqueue_many=True,
                                      batch_size=batch_size,
                                      capacity=batch_size * capacity_in_batches_train,
                                      allow_smaller_final_batch=False,
                                      num_threads=num_threads)
        else:
            print("(dataloader.py): input_type is 'image_pairs'; split is '{}'".format(split_name))
            if split_name == 'valid':
                return tf.train.batch([image_as, image_bs, flows],
                                      enqueue_many=True,
                                      batch_size=batch_size,
                                      capacity=batch_size * capacity_in_batches_val,
                                      allow_smaller_final_batch=False,
                                      num_threads=num_threads)
            else:
                return tf.train.batch([image_as, image_bs, flows],
                                      enqueue_many=True,
                                      batch_size=batch_size,
                                      capacity=batch_size * capacity_in_batches_train,
                                      allow_smaller_final_batch=False,
                                      num_threads=num_threads)
