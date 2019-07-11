# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import copy
slim = tf.contrib.slim
from math import exp
from .dataset_configs import FLYING_CHAIRS_ALL_DATASET_CONFIG, SINTEL_FINAL_ALL_DATASET_CONFIG,\
    SINTEL_ALL_DATASET_CONFIG, FLYING_THINGS_3D_ALL_DATASET_CONFIG, FC_TRAIN_SINTEL_VAL_DATASET_CONFIG,\
    FT3D_TRAIN_SINTEL_VAL_DATASET_CONFIG

_preprocessing_ops = tf.load_op_library(
    tf.resource_loader.get_path_to_datafile("./ops/build/preprocessing.so"))


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
                dataset_config == FT3D_TRAIN_SINTEL_VAL_DATASET_CONFIG:
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


# TODO: fix bug with data augmentation
def load_batch(dataset_config_str, split_name, global_step=None, input_type='image_pairs', common_queue_capacity=128,
               common_queue_min=64, capacity_in_batches_train=4, capacity_in_batches_val=1, num_threads=8,
               batch_size=None):

    if dataset_config_str.lower() == 'flying_things3d':
        dataset_config = FLYING_THINGS_3D_ALL_DATASET_CONFIG
    elif dataset_config_str.lower() == 'sintel_final':
        dataset_config = SINTEL_FINAL_ALL_DATASET_CONFIG
    elif dataset_config_str.lower() == 'sintel_all':  # clean + final
        dataset_config = SINTEL_ALL_DATASET_CONFIG
    elif dataset_config_str.lower() == 'fc_sintel':  # FC (train) + Sintel (validation)
        dataset_config = FC_TRAIN_SINTEL_VAL_DATASET_CONFIG
    elif dataset_config_str.lower() == 'ft3d_sintel':  # FT3D (train) + Sintel (validation)
        dataset_config = FT3D_TRAIN_SINTEL_VAL_DATASET_CONFIG
    else:  # flying_chairs
        dataset_config = FLYING_CHAIRS_ALL_DATASET_CONFIG

    if batch_size is not None:
        print("Batch size changed from training_schedules.py default '{}' to '{}'".format(
            dataset_config['BATCH_SIZE'], batch_size))
    else:
        print("Batch size kept to default value: {}".format(dataset_config['BATCH_SIZE']))
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
            shuffle=True,)

        if input_type == 'image_matches':
            image_b = None
            image_a, matches_a, sparse_flow, flow = data_provider.get(['image_a', 'matches_a', 'sparse_flow', 'flow'])
            # tensors are already of type float (redundant conversion), remove when everything is tested
            # image_a, matches_a, sparse_flow, flow = map(tf.to_float, [image_a, matches_a, sparse_flow, flow])
            # image_a, matches_a = map(tf.to_float, [image_a, matches_a])

            image_a, matches_a = map(lambda x: tf.cast(x, dtype=tf.float32), [image_a, matches_a])  # flow is float32
        else:
            matches_a = None
            sparse_flow = None
            image_a, image_b, flow = data_provider.get(['image_a', 'image_b', 'flow'])
            # tensors are already of type float (redundant conversion), remove when everything is tested
            # image_a, image_b, flow = map(tf.to_float, [image_a, image_b, flow])
            # Need all to be float32 for certain operations (flow is already of type float32)
            # image_a, image_b = map(tf.to_float, [image_a, image_b])
            image_a, image_b = map(lambda x: tf.cast(x, dtype=tf.float32), [image_a, image_b])

        if dataset_config['PREPROCESS']['scale']:
            image_a = image_a / 255.0
            if not input_type == 'image_matches':
                image_b = image_b / 255.0

        if dataset_config_str.lower() == 'fc_sintel' or dataset_config_str.lower() == 'ft3d_sintel':
            if split_name == 'train':
                crop = [dataset_config['PREPROCESS']['crop_height'][0],
                        dataset_config['PREPROCESS']['crop_width'][0]]
            elif split_name == 'valid':
                crop = [dataset_config['PREPROCESS']['crop_height'][1],
                        dataset_config['PREPROCESS']['crop_width'][1]]
            else:
                raise ValueError("FATAL: unexpected 'split_name'. Must be either 'train' or 'valid'")
        config_a = config_to_arrays(dataset_config['PREPROCESS']['image_a'])
        config_b = config_to_arrays(dataset_config['PREPROCESS']['image_b'])

        if input_type == 'image_matches':
            image_bs = None
            image_as, matches_as, sparse_flows, flows = map(lambda x: tf.expand_dims(x, 0),
                                                            [image_a, matches_a, sparse_flow, flow])
        else:
            matches_as = None
            sparse_flows = None
            image_as, image_bs, flows = map(lambda x: tf.expand_dims(x, 0), [image_a, image_b, flow])

        #
        # Perform data augmentation on GPU  fperezgamonal: typo, it does not work on the GPU, only on the CPU!
        # TODO: despite not reporting segmentation fault, the process is killed when data_augmentation is added. Test:
        #   - See if it works in GPU (more memory)
        #   - Test with more CPUs and/or memory per cpu (harder to get in the queue to test anything!)
        #   - Play around with queue sizes and num_threads (maybe the queue is still too large (or short?))
        # with tf.device('/cpu:0'):  # it should work on the gpu according to the test.py (repo root folder)
        #     image_as, image_bs, transforms_from_a, transforms_from_b = \
        #         _preprocessing_ops.data_augmentation(image_as,
        #                                              image_bs,
        #                                              global_step,
        #                                              crop,
        #                                              config_a['name'],
        #                                              config_a['rand_type'],
        #                                              config_a['exp'],
        #                                              config_a['mean'],
        #                                              config_a['spread'],
        #                                              config_a['prob'],
        #                                              config_a['coeff_schedule'],
        #                                              config_b['name'],
        #                                              config_b['rand_type'],
        #                                              config_b['exp'],
        #                                              config_b['mean'],
        #                                              config_b['spread'],
        #                                              config_b['prob'],
        #                                              config_b['coeff_schedule'])
        #
        #     noise_coeff_a = None
        #     noise_coeff_b = None
        #
        #     # Generate and apply noise coeff for A if defined in A params
        #     if 'noise' in dataset_config['PREPROCESS']['image_a']:
        #         discount_coeff = tf.constant(1.0)
        #         if 'coeff_schedule_param' in dataset_config['PREPROCESS']['image_a']:
        #             initial_coeff = dataset_config['PREPROCESS']['image_a']['coeff_schedule_param']['initial_coeff']
        #             final_coeff = dataset_config['PREPROCESS']['image_a']['coeff_schedule_param']['final_coeff']
        #             half_life = dataset_config['PREPROCESS']['image_a']['coeff_schedule_param']['half_life']
        #             discount_coeff = initial_coeff + \
        #                 (final_coeff - initial_coeff) * \
        #                 (2.0 / (1.0 + exp(-1.0986 * global_step / half_life)) - 1.0)
        #
        #         noise_coeff_a = _generate_coeff(
        #             dataset_config['PREPROCESS']['image_a']['noise'], discount_coeff)
        #         noise_a = tf.random_normal(shape=tf.shape(image_as),
        #                                    mean=0.0, stddev=noise_coeff_a,
        #                                    dtype=tf.float32)
        #         image_as = tf.clip_by_value(image_as + noise_a, 0.0, 1.0)
        #
        #     # Generate noise coeff for B if defined in B params
        #     if 'noise' in dataset_config['PREPROCESS']['image_b']:
        #         discount_coeff = tf.constant(1.0)
        #         if 'coeff_schedule_param' in dataset_config['PREPROCESS']['image_b']:
        #             initial_coeff = dataset_config['PREPROCESS']['image_b']['coeff_schedule_param']['initial_coeff']
        #             final_coeff = dataset_config['PREPROCESS']['image_b']['coeff_schedule_param']['final_coeff']
        #             half_life = dataset_config['PREPROCESS']['image_b']['coeff_schedule_param']['half_life']
        #             discount_coeff = initial_coeff + \
        #                 (final_coeff - initial_coeff) * \
        #                 (2.0 / (1.0 + exp(-1.0986 * global_step / half_life)) - 1.0)
        #         noise_coeff_b = _generate_coeff(
        #             dataset_config['PREPROCESS']['image_b']['noise'], discount_coeff)
        #
        #     # Combine coeff from a with coeff from b
        #     if noise_coeff_a is not None:
        #         if noise_coeff_b is not None:
        #             noise_coeff_b = noise_coeff_a * noise_coeff_b
        #         else:
        #             noise_coeff_b = noise_coeff_a
        #
        #     # Add noise to B if needed
        #     if noise_coeff_b is not None:
        #         noise_b = tf.random_normal(shape=tf.shape(image_bs),
        #                                    mean=0.0, stddev=noise_coeff_b,
        #                                    dtype=tf.float32)
        #         image_bs = tf.clip_by_value(image_bs + noise_b, 0.0, 1.0)
        #
        #     # Perform flow augmentation using spatial parameters from data augmentation
        #     flows = _preprocessing_ops.flow_augmentation(
        #         flows, transforms_from_a, transforms_from_b, crop)

        if input_type == 'image_matches':
            if split_name == 'valid':
                return tf.train.batch([image_as, matches_as, sparse_flows, flows],
                                      enqueue_many=True,
                                      batch_size=batch_size,
                                      capacity=batch_size * capacity_in_batches_val,
                                      allow_smaller_final_batch=False,
                                      num_threads=num_threads)
            else:

                return tf.train.batch([image_as, matches_as, sparse_flows, flows],
                                      enqueue_many=True,
                                      batch_size=batch_size,
                                      capacity=batch_size * capacity_in_batches_train,
                                      allow_smaller_final_batch=False,
                                      num_threads=num_threads)
        else:
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
