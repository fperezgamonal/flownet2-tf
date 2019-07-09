import abc
from enum import Enum
from math import ceil
import os
import tensorflow as tf
import numpy as np
import sys
import datetime
import uuid
from imageio import imread, imsave
from .flowlib import flow_to_image, write_flow, read_flow, compute_all_metrics, get_metrics
from .training_schedules import LONG_SCHEDULE, FINE_SCHEDULE, SHORT_SCHEDULE, FINETUNE_SINTEL_S1, FINETUNE_SINTEL_S2, \
    FINETUNE_SINTEL_S3, FINETUNE_SINTEL_S4, FINETUNE_SINTEL_S5, FINETUNE_KITTI_S1, FINETUNE_KITTI_S2,\
    FINETUNE_KITTI_S3, FINETUNE_KITTI_S4, FINETUNE_ROB, LR_RANGE_TEST, CLR_SCHEDULE, EXP_DECREASING
from .utils import exponentially_increasing_lr, exponentially_decreasing_lr, _lr_cyclic, _mom_cyclic
slim = tf.contrib.slim

VAL_INTERVAL = 1000


# The optimizer state could not be properly resumed because of the following reasons:
#   * Actual restoring from checkpoint was done BEFORE defining the graph operations==> only global_step resumed
#   * SLIM required us to use assign_from_checkpoint_fn with the vars retrieved by optimistic_restore_vars
#   * The above is passed as init_fn to learning.train()
#   * Finally, the custom training is also passed (but we have to check if its configuration is actually applied or
#       the one created by init_fn is used instead, without the max num of checkpoints, etc.)

# Special thanks to helpful discussions on similar issues:
#   * optimistic_restore_vars:
#       https://github.com/tensorflow/tensorflow/issues/312#issuecomment-335039382 (the whole discussion):
#   * 'slim.learning.train can't restore variables if new variable have created

# Important notice: tf.slim is considered deprecated so everything should be moved to tf.estimator high level API
# However, for the scope of the MsC thesis, it is too complicated to write flownet2-tf from scratch again, so we
# patch it to meet our needs despite its problems
def optimistic_restore_vars(model_checkpoint_path, reset_global_step=False):
    reader = tf.train.NewCheckpointReader(model_checkpoint_path)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                       if var.name.split(':')[0] in saved_shapes])

    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                # if reset_global_step and 'global_step' in var_name:
                #     print("Found global step with var_name: '{}'".format(var_name))
                #     # print("Removing tensor from restoring variables list...")
                #     # restore_vars.pop(-1)
                #     # Assign 0 (instead of completely removing it from varlist, as it is safer)
                #     print("Resetting its value back to 0 (first iteration)...")
                #     curr_var.assign(0)
                #     print("This should give us 0, global_step= {}".format(tf.train.get_global_step()))

                # For any var, append it to the list of variables to be restored
                restore_vars.append(curr_var)

                if reset_global_step and 'global_step' in var_name:
                    restore_vars.pop(-1)  # not ideal but 'assign' is not working for some reason

    return restore_vars


# TODO: only works if both train and validation losses are provided! (i.e. if valid_iters < 0 it fails)
# ==== Generate smooth version (EMA) of the training and validation losses ====
# Based off: https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py#L302
def _add_loss_summaries(train_loss, valid_loss, decay=0.99, summary_name_train='train/smoothed_loss',
                        summary_name_valid='valid/smoothed_loss', log_tensorboard=True):
    """Add summaries for losses in a given model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      train_loss: Total loss from training (from self.model())
      valid_loss: validation loss
      decay: decay factor
      summary_name_train: name for the smoothed training loss summary
      summary_name_valid: name for the smoothed validation loss summary
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    ema = tf.train.ExponentialMovingAverage(decay, zero_debias=True)
    loss_averages_op = ema.apply([train_loss, valid_loss])

    if log_tensorboard:
        tf.summary.scalar(summary_name_train, ema.average(train_loss))
        tf.summary.scalar(summary_name_valid, ema.average(valid_loss))

    return loss_averages_op


class Mode(Enum):
    TRAIN = 1
    TEST = 2


class Net(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, mode=Mode.TRAIN, debug=False):
        # self.global_step = slim.get_or_create_global_step() messes things up for loading previous checkpoints
        self.mode = mode
        self.debug = debug

    @abc.abstractmethod
    def model(self, inputs, training_schedule, trainable=True):
        """
        Defines the model and returns a tuple of Tensors needed for calculating the loss.
        """
        return

    @abc.abstractmethod
    def loss(self, **kwargs):
        """
        Accepts prediction Tensors from the output of `model`.
        Returns a single Tensor representing the total loss of the model.
        """
        return

    def get_training_schedule(self, training_schedule_str):
        if training_schedule_str.lower() == 'fine':  # normally applied after long on FlyingThings3D
            training_schedule = FINE_SCHEDULE
        elif training_schedule_str.lower() == 'short':  # quicker schedule to train on FlyingChairs (or any dataset)
            training_schedule = SHORT_SCHEDULE
        elif training_schedule_str.lower() == 'clr':  # cyclical learning rate
            training_schedule = CLR_SCHEDULE
        elif training_schedule_str.lower() == 'exp_decr':  # exponentially decreasing learning rate (w. min+max LR)
            training_schedule = EXP_DECREASING
        # SINTEL
        elif training_schedule_str.lower() == 'sintel_s1':  # Fine-tune Sintel (stage 1)
            training_schedule = FINETUNE_SINTEL_S1
        elif training_schedule_str.lower() == 'sintel_s2':
            training_schedule = FINETUNE_SINTEL_S2
        elif training_schedule_str.lower() == 'sintel_s3':
            training_schedule = FINETUNE_SINTEL_S3
        elif training_schedule_str.lower() == 'sintel_s4':
            training_schedule = FINETUNE_SINTEL_S4
        elif training_schedule_str.lower() == 'sintel_s5':
            training_schedule = FINETUNE_SINTEL_S5
        # KITTI
        elif training_schedule_str.lower() == 'kitti_s1':
            training_schedule = FINETUNE_KITTI_S1
        elif training_schedule_str.lower() == 'kitti_s2':
            training_schedule = FINETUNE_KITTI_S2
        elif training_schedule_str.lower() == 'kitti_s3':
            training_schedule = FINETUNE_KITTI_S3
        elif training_schedule_str.lower() == 'kitti_s4':
            training_schedule = FINETUNE_KITTI_S4
        # ROB (robust challenge: mix of Sintel, KITTI, HD1K and Middlebury
        elif training_schedule_str.lower() == 'rob':
            training_schedule = FINETUNE_ROB
        # Learning rate range test
        elif training_schedule_str.lower() == 'lr_range_test':
            training_schedule = LR_RANGE_TEST
        else:  # long schedule as default
            training_schedule = LONG_SCHEDULE  # Normally applied from scratch on FlyingChairs

        return training_schedule

    def get_training_schedule_folder_string(self, training_schedule_str):
        if 'sintel' in training_schedule_str.lower():  # put all Sintel fine-tuning under the same log folder
            training_schedule_fld = 'fine_sintel'
        elif 'fine' in training_schedule_str.lower():  # format it a little better, FT3D for FlyingThings3D
            training_schedule_fld = "Sfine_FT3D"
        elif 'clr' in training_schedule_str.lower():  # make it all caps for readability
            training_schedule_fld = "CLR"
        elif 'exp_decreasing' in training_schedule_str.lower():
            training_schedule_fld = "exp_decr"
        elif 'long' in training_schedule_str.lower():
            training_schedule_fld = "Slong_FC".format(training_schedule_str)  # FC for FlyingChairs
        else:
            training_schedule_fld = training_schedule_str  # leave it as is

        return training_schedule_fld

    # based on github.com/philferriere/tfoptflow/blob/33e8a701e34c8ce061f17297d40619afbd459ade/tfoptflow/model_pwcnet.py
    # functions: adapt_x, adapt_y, postproc_y_hat (crop)
    def adapt_x(self, input_a, input_b=None, matches_a=None, sparse_flow=None, divisor=64):
        """
        Adapts the input images/matches to the required network dimensions
        :param input_a: first image
        :param input_b: second image (optional if we use sparse flow + the first image)
        :param matches_a: (optional) matches between first and second images
        :param sparse_flow: (optional) sparse optical flow field initialised from a set of sparse matches
        :param divisor: (optional) number by which all image sizes should be divisible by. divisor=2^n_pyram, here 6
        :return: padded and normalised input_a, input_b and sparse_flow (if needed)
        """
        # Convert from RGB -> BGR
        # First + second image
        input_a = input_a[..., [2, 1, 0]]
        if sparse_flow is not None and matches_a is not None:
            matches_a = matches_a[..., np.newaxis]  # from (height, width) to (height, width, 1)
            print("New scheme: first + matchtrain_losses (1st=> 2nd frame) given")
        else:
            print("Normal scheme: first + second frame given")  # only for debugging, remove afterwards
            input_b = input_b[..., [2, 1, 0]]

        # Scale from [0, 255] -> [0.0, 1.0] if needed
        if input_a.max() > 1.0:
            input_a = input_a / 255.0
        if sparse_flow is not None and matches_a is not None:
            matches_a = matches_a / 255.0
        else:
            if input_b.max() > 1.0:
                input_b = input_b / 255.0

        height_a, width_a, channels_a = input_a.shape  # temporal hack so it works with any size (batch or not)
        if sparse_flow is not None and matches_a is not None:
            height_ma, width_ma, channels_ma = matches_a.shape
            assert height_ma == height_a and width_ma == width_a and channels_ma == 1, (
                "Mask has invalid dimensions. Should be ({0}, {1}, 1) but are {2}".format(height_a, width_a,
                                                                                          matches_a.shape)
            )
        else:
            height_b, width_b, channels_b = input_b.shape
            # Assert matching image sizes
            assert height_a == height_b and width_a == width_b and channels_a == channels_b, (
                "FATAL: image dimensions do not match. Image 1 has shape: {0}, Image 2 has shape: {1}".format(
                    input_a.shape, input_b.shape
                )
            )

        if height_a % divisor != 0 or width_a % divisor != 0:
            new_height = int(ceil(height_a / divisor) * divisor)
            new_width = int(ceil(width_a / divisor) * divisor)
            pad_height = new_height - height_a
            pad_width = new_width - width_a

            if self.mode == Mode.TRAIN:  # working with batches, adapt to match dimensions (batch, height, width, ch)
                padding = [(0, 0), (0, pad_height), (0, pad_width), (0, 0)]
            elif self.mode == Mode.TEST:  # working with pair of images (for now there is no inference on whole batches)
                # TODO: modify test.py so we can predict batches (much quicker for whole datasets) and simplify this!
                padding = [(0, pad_height), (0, pad_width), (0, 0)]
            else:
                padding = [(0, pad_height), (0, pad_width), (0, 0)]

            x_adapt_info = input_a.shape  # Save original shape
            input_a = np.pad(input_a, padding, mode='constant', constant_values=0.)

            if sparse_flow is not None and matches_a is not None:
                matches_a = np.pad(matches_a, padding, mode='constant', constant_values=0.)
                sparse_flow = np.pad(sparse_flow, padding, mode='constant', constant_values=0.)
            else:
                input_b = np.pad(input_b, padding, mode='constant', constant_values=0.)
        else:
            x_adapt_info = None

        return input_a, input_b, matches_a, sparse_flow, x_adapt_info

    # This is not used in training since we load already padded flows. If it applies, use in test for 'sparse_flow'
    def adapt_y(self, flow, divisor=64):
        """
        Adapts the ground truth flows to train to the required network dimensions
        :param flow: ground truth optical flow between first and second image
        :param divisor: divisor by which all image sizes should be divisible by. In fact: divisor=2^num_pyramids, here 6
        :return: padded ground truth optical flow
        """
        # Assert it is a valid flow
        assert flow.shape[-1] == 2
        height = flow.shape[-3]  # temporally as this to account for batch/no batch tensor dimensions (also in adapt_x)
        width = flow.shape[-2]

        if height % divisor != 0 or width % divisor != 0:
            new_height = int(ceil(height / divisor) * divisor)
            new_width = int(ceil(width / divisor) * divisor)
            pad_height = new_height - height
            pad_width = new_width - width

            if self.mode == Mode.TRAIN:  # working with batches, adapt to match dimensions
                padding = [(0, 0), (0, pad_height), (0, pad_width), (0, 0)]
            elif self.mode == Mode.TEST:  # working with pair of images (for now there is no inference on whole batches)
                # TODO: modify test.py so we can predict batches (much quicker for whole datasets) and simplify this!
                padding = [(0, pad_height), (0, pad_width), (0, 0)]
            else:
                padding = [(0, pad_height), (0, pad_width), (0, 0)]

            y_adapt_info = flow.shape  # Save original size
            flow = np.pad(flow, padding, mode='constant', constant_values=0.)
        else:
            y_adapt_info = None

        return flow, y_adapt_info

    def adapt_sample(self, sample, divisor=64):
        """
        Adapts a sample image by padding it with zeros if it is not multiple of divisor
        :param sample: image to be padded (is a tf tensor so we should use tf.shape instead of numpy's A.shape)
        :param divisor: number by which the dimensions of sample must be divisible by
        :return: the padded sample that can be fed to the network
        """
        if len(sample.get_shape()) == 4:  # batch
            batch, height, width, channels = sample.get_shape().as_list()
        elif len(tf.shape(sample)) == 3:  # standard 3 channels image
            height, width, channels = sample.get_shape().as_list()
        else:
            raise ValueError("expected a tensor with 3 or 4 dimensions but {} were given".format(len(sample.shape)))

        if height % divisor != 0 or width % divisor != 0:
            new_height = int(ceil(height / divisor) * divisor)
            new_width = int(ceil(width / divisor) * divisor)
            pad_height = new_height - height
            pad_width = new_width - width

            if self.mode == Mode.TRAIN:  # working with batches, adapt to match dimensions
                padding = [(0, 0), (0, pad_height), (0, pad_width), (0, 0)]
            elif self.mode == Mode.TEST:  # working with pair of images (for now there is no inference on whole batches)
                # TODO: modify test.py so we can predict batches (much quicker for whole datasets) and simplify this!
                padding = [(0, pad_height), (0, pad_width), (0, 0)]
            else:
                padding = [(0, pad_height), (0, pad_width), (0, 0)]

            sample_adapt_info = sample.get_shape().as_list()  # Save original size
            sample = np.pad(sample, padding, mode='constant', constant_values=0.)
        else:
            sample_adapt_info = None

        return sample, sample_adapt_info

    def postproc_y_hat_test(self, pred_flows, adapt_info=None):
        """
        Postprocess the output flows during test mode
        :param pred_flows: predictions
        :param adapt_info: None if input image is multiple of 'divisor', original size otherwise
        :return: post-processed flow (cropped to original size if need be)
        """
        if adapt_info is not None:  # must define it when padding!
            # pred_flows = pred_flows[:, 0:adapt_info[1], 0:adapt_info[2], :]  # batch!
            pred_flows = pred_flows[0:adapt_info[-3], 0:adapt_info[-2], :]  # one sample
        return pred_flows

    def postproc_y_hat_train(self, y_hat):
        """
        Postprocess the output flows during train mode
        :param y_hat: predictions
        :return: batch loss and metric
        """

        return y_hat[0], y_hat[1]

    def test(self, checkpoint, input_a_path, input_b_path=None, matches_a_path=None, sparse_flow_path=None,
             out_path='./', input_type='image_pairs', save_image=True, save_flo=True, compute_metrics=True,
             gt_flow=None, occ_mask=None, inv_mask=None):
        """

        :param checkpoint:
        :param input_a_path:
        :param input_b_path:
        :param matches_a_path:
        :param sparse_flow_path:
        :param out_path:
        :param input_type:
        :param save_image:
        :param save_flo:
        :param compute_metrics:
        :param gt_flow:
        :param occ_mask:
        :param inv_mask:
        :return:
        """
        input_a = imread(input_a_path)

        if sparse_flow_path is not None and matches_a_path is not None and input_type == 'image_matches':
            # Read matches mask and sparse flow from file
            input_b = None
            matches_a = imread(matches_a_path)
            sparse_flow = read_flow(sparse_flow_path)
            assert sparse_flow.shape[-1] == 2  # assert it is a valid flow
        else:  # Define them as None (although in 'apply_x' they default to None, JIC)
            input_b = imread(input_b_path)
            sparse_flow = None
            matches_a = None
        input_a, input_b, matches_a, sparse_flow, x_adapt_info = self.adapt_x(input_a, input_b,
                                                                              matches_a, sparse_flow)

        # TODO: This is a hack, we should get rid of this
        # the author probably means that it should be chosen as an input parameter not hardcoded!
        training_schedule = LONG_SCHEDULE

        if sparse_flow_path is None:
            inputs = {
                'input_a': tf.expand_dims(tf.constant(input_a, dtype=tf.float32), 0),
                # uint8 may cause mismatch format when concatenating (after normalisation we
                # probably have a float anyway (due to the .0 in 255.0))
                'input_b': tf.expand_dims(tf.constant(input_b, dtype=tf.float32), 0),
            }
        else:
            inputs = {
                'input_a': tf.expand_dims(tf.constant(input_a, dtype=tf.float32), 0),
                # uint8 may cause mismatch format when concatenating (after normalisation we
                # probably have a float anyway (due to the .0 in 255.0))
                'matches_a': tf.expand_dims(tf.constant(matches_a, dtype=tf.float32), 0),
                'sparse_flow': tf.expand_dims(tf.constant(sparse_flow, dtype=tf.float32), 0),
            }
        predictions = self.model(inputs, training_schedule)
        pred_flow = predictions['flow']

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            pred_flow = sess.run(pred_flow)[0, :, :, :]
            if x_adapt_info is not None:
                y_adapt_info = (x_adapt_info[-3], x_adapt_info[-2], 2)
            else:
                y_adapt_info = None
            pred_flow = self.postproc_y_hat_test(pred_flow, adapt_info=y_adapt_info)

            # unique_name = 'flow-' + str(uuid.uuid4())  completely random and not useful to evaluate metrics after!
            unique_name = input_a_path.split('/')[-1][:-4]

            if save_image or save_flo:
                if not os.path.isdir(out_path):
                    os.makedirs(out_path)

            if save_image:
                flow_img = flow_to_image(pred_flow)
                full_out_path = os.path.join(out_path, unique_name + '_viz.png')
                imsave(full_out_path, flow_img)

            if save_flo:
                full_out_path = os.path.join(out_path, unique_name + '_flow.flo')
                write_flow(pred_flow, full_out_path)

            if compute_metrics and gt_flow is not None:
                gt_flow = read_flow(gt_flow)
                if occ_mask is not None:
                    occ_mask = imread(occ_mask)
                if inv_mask is not None:
                    inv_mask = imread(inv_mask)

                # Compute all metrics
                metrics = compute_all_metrics(pred_flow, gt_flow, occ_mask=occ_mask, inv_mask=inv_mask)
                final_str_formated = get_metrics(metrics)
                print(final_str_formated)

    # TODO: double-check the number of columns of the txt file to ensure it is properly formatted
    # Each line will define a set of inputs. By doing so, we reduce complexity and avoid errors due to "forced" sorting
    def test_batch(self, checkpoint, image_paths, out_path, input_type='image_pairs', save_image=True, save_flo=True,
                   compute_metrics=True, log_metrics2file=False):
        """
        Run inference on a set of images defined in .txt files
        :param checkpoint: the path to the pre-trained model weights
        :param image_paths: path to the txt file where the image paths are listed
        :param out_path: output path where flows and visualizations should be stored
        :param input_type: whether we are dealing with two consecutive frames or one frame + matches (interpolation)
        :param save_image: whether to save a png visualization (Middlebury colour code) of the flow
        :param save_flo: whether to save the 'raw' .flo file (useful to compute errors and such)
        :param compute_metrics: whether to compute error metrics or not
        :param log_metrics2file: whether to log the metrics to a file instead of printing them to stdout
        :return:
        """
        # Build Graph
        input_a = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3])

        if input_type == 'image_matches':
            matches_a = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 1])
            sparse_flow = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 2])
            inputs = {
                'input_a': input_a,
                'matches_a': matches_a,
                'sparse_flow': sparse_flow,
            }
        else:
            input_b = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
            inputs = {
                'input_a': input_a,
                'input_b': input_b,
            }

        training_schedule = LONG_SCHEDULE  # any would suffice, as it is not used (??)
        predictions = self.model(inputs, training_schedule, trainable=False)
        pred_flow = predictions['flow']

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            # Read and process the resulting list, one element at a time
            with open(image_paths, 'r') as input_file:
                path_list = input_file.read()

            if log_metrics2file:
                basefile = image_paths.split()[-1]
                logfile = basefile.replace('.txt', '_metrics.log')
                logfile = open(logfile, 'w+')

            for img_idx in range(len(path_list)):
                # Read + pre-process files
                # Each line is split into a list with N elements (separator: blank space (" "))
                path_inputs = path_list[img_idx].split(' ')
                assert 2 <= len(path_inputs) <= 6, (
                    'More paths than expected. Expected: I1+I2 (2), I1+MM+SF(3), I1+MM+SF+GTF(4),'
                    '  I1+MM+SF+GT+OCC_MSK+INVMASK(5 to 6)')

                if len(path_inputs) == 2 and input_type == 'image_pairs':  # Only image1 + image2 have been provided
                    frame_0 = imread(path_inputs[0])
                    frame_1 = imread(path_inputs[1])
                    matches_0 = None
                    sparse_flow_0 = None
                elif len(path_inputs) == 3 and input_type == 'image_matches':  # image1 + matches mask + sparse_flow
                    frame_0 = imread(path_inputs[0])
                    frame_1 = None
                    matches_0 = imread(path_inputs[1])
                    sparse_flow_0 = read_flow(path_inputs[2])

                # image1 + image2 + ground truth flow
                elif len(path_inputs) == 3 and input_type == 'image_pairs':
                    frame_0 = imread(path_inputs[0])
                    frame_1 = imread(path_inputs[1])
                    matches_0 = None
                    sparse_flow_0 = None
                    gt_flow_0 = read_flow(path_inputs[2])

                # img1 + matches + sparse + gt flow
                elif len(path_inputs) == 4 and input_type == 'image_matches':
                    frame_0 = imread(path_inputs[0])
                    frame_1 = None
                    matches_0 = imread(path_inputs[1])
                    sparse_flow_0 = read_flow(path_inputs[2])
                    gt_flow_0 = read_flow(path_inputs[3])

                # img1 + img2 + gtflow + occ_mask
                elif len(path_inputs) == 4 and input_type == 'image_pairs' and compute_metrics:
                    frame_0 = imread(path_inputs[0])
                    frame_1 = imread(path_inputs[1])
                    matches_0 = None
                    sparse_flow_0 = None
                    gt_flow_0 = read_flow(path_inputs[2])
                    occ_mask_0 = imread(path_inputs[3])

                # img1 + img2 + gtflow + occ_mask + inv_mask
                elif len(path_inputs) == 5 and input_type == 'image_pairs' and compute_metrics:
                    frame_0 = imread(path_inputs[0])
                    frame_1 = imread(path_inputs[1])
                    matches_0 = None
                    sparse_flow_0 = None
                    gt_flow_0 = read_flow(path_inputs[2])
                    occ_mask_0 = imread(path_inputs[3])
                    inv_mask_0 = imread(path_inputs[4])

                # img1 + mtch + spflow + gt_flow + occ_mask
                elif len(path_inputs) == 5 and input_type == 'image_matches' and compute_metrics:
                    frame_0 = imread(path_inputs[0])
                    frame_1 = None
                    matches_0 = imread(path_inputs[1])
                    sparse_flow_0 = read_flow(path_inputs[2])
                    gt_flow_0 = read_flow(path_inputs[3])
                    occ_mask_0 = imread(path_inputs[4])

                # img1 + mtch + spflow + gt_flow + occ_mask + inv_mask
                elif len(path_inputs) == 6 and input_type == 'image_matches' and compute_metrics:
                    frame_0 = imread(path_inputs[0])
                    frame_1 = None
                    matches_0 = imread(path_inputs[1])
                    sparse_flow_0 = read_flow(path_inputs[2])
                    gt_flow_0 = read_flow(path_inputs[3])
                    occ_mask_0 = imread(path_inputs[4])
                    inv_mask_0 = imread(path_inputs[5])

                # img1 + img2 + spflow + gtflow + occ_mask
                else:  # path to all inputs (read all and decide what to use based on 'input_type')
                    frame_0 = imread(path_inputs[0])
                    frame_1 = imread(path_inputs[1])
                    matches_0 = imread(path_inputs[2])
                    sparse_flow_0 = read_flow(path_inputs[3])
                    gt_flow_0 = read_flow(path_inputs[4])
                    occ_mask_0 = imread(path_inputs[5])
                    inv_mask_0 = imread(path_inputs[6])

                frame_0, frame_1, matches_0, sparse_flow_0, x_adapt_info = self.adapt_x(frame_0, frame_1, matches_0,
                                                                                        sparse_flow_0)
                if sparse_flow is not None and matches_0 is not None and input_type == 'image_matches':
                    flow = sess.run(pred_flow, feed_dict={
                        input_a: frame_0, input_b: frame_1,
                        matches_a: matches_0, sparse_flow: sparse_flow_0,
                    })[0, :, :, :]
                else:
                    flow = sess.run(pred_flow, feed_dict={
                        input_a: frame_0, input_b: frame_1,
                    })[0, :, :, :]

                if x_adapt_info is not None:
                    y_adapt_info = (x_adapt_info[-3], x_adapt_info[-2], 2)
                else:
                    y_adapt_info = None
                pred_flow = self.postproc_y_hat_test(flow, adapt_info=y_adapt_info)

                # unique_name = 'flow-' + str(uuid.uuid4())  completely random and not useful to evaluate metrics after!
                # TODO: modify to keep the folder structure (at least parent folder of the image) ==> test!
                # Assumption: take as reference the parent folder of the first image (common to all input types)
                # Same for the name of the output flow (take the first image name)
                parent_folder_name = path_inputs[0].split('/')[-2]
                unique_name = path_inputs[0].split('/')[-1][:-4]
                out_path = os.path.join(out_path, parent_folder_name)

                if save_image or save_flo:
                    if not os.path.isdir(out_path):
                        os.makedirs(out_path)

                if save_image:
                    flow_img = flow_to_image(pred_flow)
                    full_out_path = os.path.join(out_path, unique_name + '_viz.png')
                    imsave(full_out_path, flow_img)

                if save_flo:
                    full_out_path = os.path.join(out_path, unique_name + '_flow.flo')
                    write_flow(pred_flow, full_out_path)

                if compute_metrics:
                    # Compute all metrics
                    metrics = compute_all_metrics(pred_flow, gt_flow_0, occ_mask=occ_mask_0, inv_mask=inv_mask_0)
                    final_str_formated = get_metrics(metrics)

                    if log_metrics2file:
                        logfile.write(final_str_formated)
                    else:  # print to stdout
                        print(final_str_formated)

            if log_metrics2file:
                logfile.close()

    def train(self, log_dir, training_schedule_str, input_a, gt_flow, input_b=None, matches_a=None, sparse_flow=None,
              valid_iters=VAL_INTERVAL, val_input_a=None, val_gt_flow=None, val_input_b=None, val_matches_a=None,
              val_sparse_flow=None, checkpoints=None, input_type='image_pairs', log_verbosity=1, log_tensorboard=True,
              lr_range_test=False, train_params_dict=None, log_smoothed_loss=True, reset_global_step=False,
              summarise_grads=False):

        """
        runs training on the network from which this method is called.
        :param log_dir:
        :param training_schedule_str:
        :param input_a:
        :param gt_flow:
        :param input_b:
        :param matches_a:
        :param sparse_flow:
        :param valid_iters:
        :param val_input_a:
        :param val_gt_flow:
        :param val_input_b:
        :param val_matches_a:
        :param val_sparse_flow:
        :param checkpoints:
        :param input_type:
        :param log_verbosity:
        :param log_tensorboard:
        :param lr_range_test:
        :param train_params_dict:
        :param log_smoothed_loss:
        :param reset_global_step:
        :param summarise_grads:

        :return:
        """
        if log_verbosity <= 1:  # print loss and tfinfo to stdout
            tf.logging.set_verbosity(tf.logging.INFO)
        else:  # debug info (more verbose)
            tf.logging.set_verbosity(tf.logging.DEBUG)

        training_schedule = self.get_training_schedule(training_schedule_str)
        if log_tensorboard:
            tf.summary.image("train/image_a", input_a, max_outputs=1)
            # Temporal to check we properly shuffle training and validation batches
            if log_verbosity > 2:
                tf.summary.scalar('debug/image_a_mean', tf.reduce_mean(input_a))
                tf.summary.scalar('debug/matches_a_mean', tf.reduce_mean(matches_a))
                tf.summary.scalar('debug/sparse_a_mean', tf.reduce_mean(sparse_flow))
                tf.summary.scalar('debug/gtflow_a_mean', tf.reduce_mean(gt_flow))
            if valid_iters > 0:
                tf.summary.image("valid/image_a", val_input_a, max_outputs=1)

                if log_verbosity > 2:
                    tf.summary.scalar('debug/val_image_a_mean', tf.reduce_mean(val_input_a))
                    tf.summary.scalar('debug/val_matches_a_mean', tf.reduce_mean(val_matches_a))
                    tf.summary.scalar('debug/val_sparse_a_mean', tf.reduce_mean(val_sparse_flow))
                    tf.summary.scalar('debug/val_gtflow_a_mean', tf.reduce_mean(val_gt_flow))

            if matches_a is not None and sparse_flow is not None and input_type == 'image_matches':
                tf.summary.image("train/matches_a", matches_a, max_outputs=1)
                if valid_iters > 0:
                    tf.summary.image("valid/matches_a", val_matches_a, max_outputs=1)
            else:
                tf.summary.image("train/image_b", input_b, max_outputs=1)
                if valid_iters > 0:
                    tf.summary.image("valid/image_b", val_input_b, max_outputs=1)

        # Initialise global step by parsing checkpoint filename to define learning rate (restoring is done afterwards)
        if checkpoints is not None:
            # Create the initial assignment op
            if isinstance(checkpoints, dict):
                for (checkpoint_path, (scope, new_scope)) in checkpoints.items():
                    variables_to_restore = slim.get_variables(scope=scope)
                    renamed_variables = {
                        var.op.name.split(new_scope + '/')[1]: var
                        for var in variables_to_restore
                    }
                # Initialise checkpoint for stacked nets with the global step as the number of the outermost net
                step_number = int(checkpoint_path.split('-')[-1])
                checkpoint_global_step_tensor = tf.Variable(step_number, trainable=False, name='global_step',
                                                            dtype='int64')
            # TODO: adapt resuming from saver to stacked architectures
            elif isinstance(checkpoints, str):
                checkpoint_path = checkpoints
                if not reset_global_step:
                    step_number = int(checkpoint_path.split('-')[-1])
                    if log_verbosity > 1:
                        print("Defining global step by parsing model filename...")
                        print("Found step number: {}".format(step_number))

                    checkpoint_global_step_tensor = tf.Variable(step_number, trainable=False, name='global_step',
                                                                dtype='int64')
                else:
                    if log_verbosity > 1:
                        print("Defining global step as 0 (reset_global_step = True)")
                    checkpoint_global_step_tensor = tf.Variable(0, trainable=False, name='global_step', dtype='int64')
            else:
                raise ValueError("checkpoint should be a single path (string) or a dictionary for stacked networks")
        else:
            checkpoint_global_step_tensor = tf.Variable(0, trainable=False, name='global_step', dtype='int64')

        # max_steps overrides max_iter which is configured in training_schedules.py
        if 'max_steps' in train_params_dict:
            training_schedule['max_iters'] = train_params_dict['max_steps']
            print("Overwritten default value of max_iters={} by user-inputted={}".format(
                training_schedule['max_iters'], train_params_dict['max_steps']))

        if lr_range_test:  # learning rate range test to bound max/min optimal learning rate (2015, Leslie N. Smith)
            if lr_range_test is not None:  # use the input params
                start_lr = train_params_dict['start_lr']
                end_lr = train_params_dict['end_lr']
                lr_range_niters = train_params_dict['lr_range_niters']

            else:  # use default values
                start_lr = 1e-10
                end_lr = 1e-1
                lr_range_niters = 10000
            if log_verbosity > 1:
                print("Learning range test config for mode '{}'".format(train_params_dict['lr_range_mode'].lower()))

            if train_params_dict['lr_range_mode'].lower() == 'exponential':
                if log_verbosity:
                    print("max_iters: {}, min_lr: {:.4f}, max_lr: {:.4f}".format(training_schedule['max_iters'],
                                                                                 start_lr, end_lr))
                learning_rate = exponentially_increasing_lr(checkpoint_global_step_tensor, min_lr=start_lr,
                                                            max_lr=end_lr, num_iters=training_schedule['max_iters'])
            else:  # linear
                if log_verbosity > 1:
                    print("===== Linear LR range test parameters =====")
                    print("base learning rate: {}, maximum learning rate: {}, step_size: {}, max_iters: {}".format(
                        start_lr, end_lr, lr_range_niters, train_params_dict['max_steps']))

                learning_rate = _lr_cyclic(g_step_op=checkpoint_global_step_tensor, base_lr=start_lr, max_lr=end_lr,
                                           step_size=lr_range_niters, mode='triangular')

        elif isinstance(training_schedule['learning_rates'], str) and not lr_range_test:  # non-piecewise learning
            # cyclical learning rate forked from: https://github.com/mhmoodlan/cyclic-learning-rate
            if training_schedule['learning_rates'].lower() == 'clr' and training_schedule_str == 'clr':
                if log_verbosity > 1:
                    print("Learning rate policy is CLR (Cyclical Learning Rate)")
                learning_rate = _lr_cyclic(
                    g_step_op=checkpoint_global_step_tensor, base_lr=train_params_dict['clr_min_lr'],
                    max_lr=train_params_dict['clr_max_lr'], step_size=train_params_dict['clr_stepsize'],
                    gamma=train_params_dict['clr_gamma'], mode=train_params_dict['clr_mode'])
                # Change max_iter according to the number of cycles and stepsize
                new_max_iters = 2 * train_params_dict['clr_stepsize'] * train_params_dict['clr_num_cycles']

                if log_verbosity > 1:
                    print("(CLR) Default max. number of iters being changed from {} to {}".format(
                        training_schedule['max_iters'], new_max_iters))
                training_schedule['max_iters'] = new_max_iters
            elif training_schedule['learning_rates'].lower() == 'exp_decr' and training_schedule_str == 'exp_decr':
                if log_verbosity > 1:
                    print("Training schedule is 'exp_decr' (exponentially decreasing LR)")
                learning_rate = exponentially_decreasing_lr(
                    checkpoint_global_step_tensor, min_lr=train_params_dict['end_lr'],
                    max_lr=train_params_dict['start_lr'], num_iters=training_schedule['max_iters'])
            else:
                if log_verbosity > 1:
                    print("Reverting to original learning rate, fixed to 3e-4 (Adam)")
                learning_rate = 3e-4  # for Adam only!
            # add other policies (1-cycle), cosine-decay, etc.
        else:
            if log_verbosity > 1:
                print("Piecewise constant learning selected (defined in 'training_schedules.py')")
            learning_rate = tf.train.piecewise_constant(checkpoint_global_step_tensor,
                                                        [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
                                                        training_schedule['learning_rates'])
        # TODO: define common variables outside of individual if-statements to keep it as short as possible
        if train_params_dict['optimizer'] is not None:
            # Stochastic Gradient Descent (SGD)
            if train_params_dict['optimizer'].lower() == 'sgd':
                if log_verbosity > 1:
                    print("Optimizer is 'SGD' (Stochastic Gradient Descent)")
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # Momentum (SGD + Momentum)
            elif train_params_dict['optimizer'].lower() == 'momentum':
                if log_verbosity > 1:
                    print("Optimizer is 'Momentum' (SGD + Momentum). Nesterov defaults to True")
                # Overides default in config
                if train_params_dict['weight_decay'] is not None:
                    training_schedule['l2_regularization'] = train_params_dict['weight_decay']
                # Use cyclic momentum if using CLR or 1 cycle (accelerates convergence)
                if ((training_schedule['learning_rates'].lower() == 'clr' and training_schedule_str == 'clr') or
                    (training_schedule['learning_rates'].lower() == 'one_cycle' and
                     (training_schedule_str == 'one_cycle')) and train_params_dict['min_momentum'] is not None and
                        train_params_dict['max_momentum'] is not None):
                    if log_verbosity > 1:
                        print("Momentum optimizer used together with CLR/1cycle, will be using cyclical momentum")
                    momentum = _mom_cyclic(g_step_op=checkpoint_global_step_tensor,
                                           base_mom=train_params_dict['min_momentum'],
                                           max_mom=train_params_dict['max_momentum'],
                                           gamma=train_params_dict['clr_gamma'],
                                           step_size=train_params_dict['clr_stepsize'], mode='triangular')

                else:  # Use fixed momentum
                    if train_params_dict['momentum'] is None:
                        if train_params_dict['max_momentum'] is not None and \
                                train_params_dict['min_momentum'] is not None and lr_range_test:
                            if log_verbosity > 1:
                                print("Using cyclical momentum (only decreasing) for LR range test...")
                                print("Cycle boundaries are: max={}, min={}".format(train_params_dict['max_momentum'],
                                                                                    train_params_dict['min_momentum']))
                            # Use cyclical momentum (only decreasing half-cycle while LR increases)
                            momentum = _lr_cyclic(g_step_op=checkpoint_global_step_tensor,
                                                  base_lr=train_params_dict['max_momentum'],
                                                  max_lr=train_params_dict['min_momentum'],
                                                  step_size=lr_range_niters,
                                                  mode='triangular')
                        else:
                            if log_verbosity > 1:
                                print("Using fixed momentum = 0.9")
                            momentum = 0.9
                    else:
                        if log_verbosity > 1:
                            print("Using user-specified (fixed) momentum value")
                        momentum = train_params_dict['momentum']

                # Track momentum value (just for debugging initially)
                if log_verbosity > 1 and log_tensorboard:
                    print("Logging cyclic momentum to tensorboard...")
                    tf.summary.scalar('cyclic_momentum', momentum)

                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
            # AdamW (w. proper weight decay not L2 regularisation), as suggested in https://arxiv.org/abs/1711.05101
            elif train_params_dict['optimizer'].lower() == 'adamw':
                if log_verbosity > 1:
                    print("Optimizer is 'AdamW' (Adam with proper Weight Decay)")
                if train_params_dict['weight_decay'] is not None:
                    weight_decay = train_params_dict['weight_decay']
                else:
                    weight_decay = 1e-4  # some reasonable default
                # adam_wd is a new class
                adam_wd = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
                # Create a adam_wd object
                optimizer = adam_wd(weight_decay=weight_decay, learning_rate=learning_rate,
                                    beta1=training_schedule['momentum'], beta2=training_schedule['momentum2'])
            else:
                # default to Adam
                if log_verbosity > 1:
                    print("Using default optimizer (Adam)")
                if training_schedule['momentum'] is None or training_schedule['momentum2'] is None:
                    training_schedule['momentum'] = 0.9
                    training_schedule['momentum2'] = 0.999

                optimizer = tf.train.AdamOptimizer(learning_rate, training_schedule['momentum'],
                                                   training_schedule['momentum2'])
        else:  # default to Adam
            if train_params_dict['weight_decay'] is not None:
                training_schedule['l2_regularization'] = train_params_dict['weight_decay']
            if training_schedule['momentum'] is None or training_schedule['momentum2'] is None:
                training_schedule['momentum'] = 0.9
                training_schedule['momentum2'] = 0.999
            optimizer = tf.train.AdamOptimizer(learning_rate, training_schedule['momentum'],
                                               training_schedule['momentum2'])

        if log_tensorboard:
            print("Logging 'learning_rate' to Tensorboard, with tensor name: '{}'".format(learning_rate.name))
            tf.summary.scalar('learning_rate', learning_rate)  # Add learning rate

        # Define input dictionary (TRAIN)
        if matches_a is not None and sparse_flow is not None and input_type == 'image_matches':
            inputs = {'input_a': input_a, 'matches_a': matches_a, 'sparse_flow': sparse_flow, }
        else:
            inputs = {'input_a': input_a, 'input_b': input_b, }
        # Define input dictionary (VALIDATION)
        if valid_iters > 0:
            if val_matches_a is not None and val_sparse_flow is not None and input_type == 'image_matches':
                val_inputs = {'input_a': val_input_a, 'matches_a': val_matches_a, 'sparse_flow': val_sparse_flow, }
            else:
                val_inputs = {'input_a': val_input_a, 'input_b': val_input_b, }

        # Define model operations (graph) to compute loss (TRAIN)
        if log_verbosity > 1:
            print("l2 regularization: {}".format(training_schedule['l2_regularization']))
        predictions = self.model(inputs, training_schedule)
        if valid_iters > 0:
            # Define model operations (graph) to compute loss (VALIDATION)
            val_predictions = self.model(val_inputs, training_schedule, trainable=False)  # test does not specify this

        # Compute losses
        train_loss = self.loss(gt_flow, predictions)
        if log_verbosity > 1:
            print("\n >>> train_loss=", train_loss)
        if valid_iters > 0:
            val_loss = self.loss(val_gt_flow, val_predictions)
            # Add validation loss to a different collection to avoid adding it to the train one when calling get_loss()
            # By default, all losses are added to the same collection (tf.GraphKeys.LOSSES)
            tf.losses.add_loss(val_loss, loss_collection='validation_losses')

        if log_tensorboard:
            summary_loss = tf.summary.scalar('train/loss', train_loss)
            # Show the generated flow in TensorBoard
            if 'flow' in predictions:
                pred_flow_0 = predictions['flow'][0, :, :, :]
                pred_flow_0 = tf.py_func(flow_to_image, [pred_flow_0], tf.uint8)
                # pred_flow_0 = tf.py_function(func=flow_to_image, inp=[pred_flow_0], Tout=tf.uint8)
                pred_flow_1 = predictions['flow'][1, :, :, :]
                pred_flow_1 = tf.py_func(flow_to_image, [pred_flow_1], tf.uint8)
                # pred_flow_1 = tf.py_function(func=flow_to_image, inp=[pred_flow_1], Tout=tf.uint8)
                pred_flow_img = tf.stack([pred_flow_0, pred_flow_1], 0)
                tf.summary.image('train/pred_flow', pred_flow_img, max_outputs=1)

            # Add ground truth flow (TRAIN)
            true_flow_0 = gt_flow[0, :, :, :]
            true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
            # true_flow_0 = tf.py_function(func=flow_to_image, inp=[true_flow_0], Tout=tf.uint8)
            true_flow_1 = gt_flow[1, :, :, :]
            true_flow_1 = tf.py_func(flow_to_image, [true_flow_1], tf.uint8)
            # true_flow_1 = tf.py_function(func=flow_to_image, inp=[true_flow_1], Tout=tf.uint8)
            true_flow_img = tf.stack([true_flow_0, true_flow_1], 0)
            tf.summary.image('train/true_flow', true_flow_img, max_outputs=1)

            # Validation
            if valid_iters > 0:
                summary_validation_loss = tf.summary.scalar('valid/loss', val_loss)
                summary_validation_delta = tf.summary.scalar("valid/loss_delta", (val_loss - train_loss))
                # Just to check that val losses are logged separately
                if log_verbosity > 1:
                    print("\n >>> validation losses=", tf.losses.get_losses(loss_collection="validation_losses"))

                # Show the generated flow in TensorBoard
                if 'flow' in val_predictions:
                    val_pred_flow_0 = val_predictions['flow'][0, :, :, :]
                    val_pred_flow_0 = tf.py_func(flow_to_image, [val_pred_flow_0], tf.uint8)
                    # val_pred_flow_0 = tf.py_function(func=flow_to_image, inp=[val_pred_flow_0], Tout=tf.uint8)
                    val_pred_flow_1 = val_predictions['flow'][1, :, :, :]
                    val_pred_flow_1 = tf.py_func(flow_to_image, [val_pred_flow_1], tf.uint8)
                    # val_pred_flow_1 = tf.py_function(func=flow_to_image, inp=[val_pred_flow_1], Tout=tf.uint8)
                    val_pred_flow_img = tf.stack([val_pred_flow_0, val_pred_flow_1], 0)
                    tf.summary.image('valid/pred_flow', val_pred_flow_img, max_outputs=1)

                # Add ground truth flow (VALIDATION)
                val_true_flow_0 = val_gt_flow[0, :, :, :]
                val_true_flow_0 = tf.py_func(flow_to_image, [val_true_flow_0], tf.uint8)
                # val_true_flow_0 = tf.py_function(func=flow_to_image, inp=[val_true_flow_0], Tout=tf.uint8)
                val_true_flow_1 = val_gt_flow[1, :, :, :]
                val_true_flow_1 = tf.py_func(flow_to_image, [val_true_flow_1], tf.uint8)
                # val_true_flow_1 = tf.py_function(func=flow_to_image, inp=[val_true_flow_1], Tout=tf.uint8)
                val_true_flow_img = tf.stack([val_true_flow_0, val_true_flow_1], 0)
                tf.summary.image('valid/true_flow', val_true_flow_img, max_outputs=1)

        # Log smoothed loss (EMA, see '_add_loss_summaries' for more details)
        if log_smoothed_loss:
            decay_factor = 0.99
            # Add exponentially moving averages for losses
            losses_average_op = _add_loss_summaries(train_loss, val_loss, decay=decay_factor,
                                                    log_tensorboard=log_tensorboard)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, losses_average_op)

        if reset_global_step:
            print("global_step has value: {}".format(tf.train.get_global_step()))
            checkpoint_global_step_tensor.assign(0)
        # Create the train_op
        training_op = slim.learning.create_train_op(
            train_loss,
            optimizer,
            summarize_gradients=summarise_grads,
            global_step=checkpoint_global_step_tensor,
        )

        # ==== Add validation by defining a custom train_step_fn ====
        # How to run validation on the same session that training (tf.slim)
        # From: https://colab.research.google.com/drive/11PWvXR85NAIe6LAV1kocheXGDJa1VOa1#scrollTo=IhwzhDFttEoh
        def train_step_fn(sess, train_op, global_step, train_step_kwargs):
            """
            slim.learning.train_step():
              train_step_kwargs = {summary_writer:, should_log:, should_stop:}

            usage: slim.learning.train( train_op, logdir,
                                        train_step_fn=train_step_fn,)
            """
            if hasattr(train_step_fn, 'step'):
                train_step_fn.step += 1  # or use global_step.eval(session=sess)
            else:
                train_step_fn.step = global_step.eval(sess)

            # calc training losses
            total_loss, should_stop = slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)

            # validate on interval (added condition to avoid evaluating on first step)
            if valid_iters > 0 and (train_step_fn.step % valid_iters == 0) and train_step_fn.step > 0:
                valid_loss, valid_delta = sess.run([val_loss, summary_validation_delta])
                print(">>> global step= {:<5d} | train={:^15.4f} | validation={:^15.4f} | delta={:^15.4f} <<<".format(
                    train_step_fn.step, total_loss, valid_loss, valid_loss - total_loss))

            return [total_loss, should_stop]
        # ===========================================================

        if checkpoints is not None:
            # Create the initial assignment op
            if isinstance(checkpoints, dict):
                for (checkpoint_path, (scope, new_scope)) in checkpoints.items():
                    variables_to_restore = slim.get_variables(scope=scope)
                    renamed_variables = {
                        var.op.name.split(new_scope + '/')[1]: var
                        for var in variables_to_restore
                    }
                    if log_verbosity > 1:
                        print("Restoring the following variables from checkpoint:")
                        for var in renamed_variables:
                            print(var)
                        print("Finished printing list of restored variables")

                # Create an initial assignment function.
                def InitAssignFn(sess):
                    sess.run(init_assign_op, init_feed_dict)

                init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, renamed_variables)
                # Initialise checkpoint for stacked nets with the global step as the number of the outermost net
                step_number = int(checkpoint_path.split('-')[-1])
                checkpoint_global_step_tensor = tf.Variable(step_number, trainable=False, name='global_step',
                                                            dtype='int64')
                # TODO: adapt resuming from saver to stacked architectures
                saver = None
            # TODO: double-check but it seems that if we remove the last checkpoint and keep and older point this fails
            # It can be patched by renaming the desired checkpoint to the name of the latest one
            # This happens because get checkpoint state searches for the latest filename in the folder so
            # ckpt.checkpoint_path keeps the old name, not updating to the "new" latest or the one specified by user
            # This behaviour is not desired when fine-tuning from a checkpoint which is not the latest (very frequent!)
            elif isinstance(checkpoints, str):
                checkpoint_path = checkpoints

                # Get checkpoint state from checkpoint_path (used to restore vars)
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_path))
                if log_verbosity > 1:
                    # print("last_ckpt_name: '{}'".format(last_ckpt_name))
                    print("ckpt.model_checkpoint_path: '{}'".format(ckpt.model_checkpoint_path))
                    print("reset_global_step: {}".format(reset_global_step))

                vars2restore = optimistic_restore_vars(ckpt.model_checkpoint_path, reset_global_step=reset_global_step)
                if log_verbosity > 1:
                    print("Path from which checkpoint state is computed: '{}'".format(os.path.dirname(checkpoint_path)))
                    print("Listing variables that will be restored(optimistic_restore_vars), total: {}:".format(
                        len(vars2restore)))
                    for var in vars2restore:
                        print(var)

                saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=2,
                                       var_list=vars2restore if checkpoint_path else None)
                # Define init function to assign variables from checkpoint
                init_fn = tf.contrib.framework.assign_from_checkpoint_fn(checkpoint_path, vars2restore)

            else:
                raise ValueError("checkpoint should be a single path (string) or a dictionary for stacked networks")
        else:
            saver = None

        # Create unique logging dir to avoid overwritting of old data (e.g.: when comparing different runs)
        now = datetime.datetime.now()
        date_now = now.strftime('%d-%m-%y_%H-%M-%S')
        training_schedule_fld = self.get_training_schedule_folder_string(training_schedule_str)

        log_dir = os.path.join(log_dir, training_schedule_fld, date_now)

        # Flush all prints (get stuck for some reason)
        sys.stdout.flush()

        print("Starting training...")
        if self.debug:
            debug_logdir = os.path.join(log_dir, 'debug')
            if not os.path.isdir(debug_logdir):
                os.makedirs(debug_logdir)
                print("debugging logdir is: {}".format(debug_logdir))
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(session)
                slim.learning.train_step(
                    session,
                    training_op,
                    checkpoint_global_step_tensor,
                    {'should_trace': tf.constant(1), 'should_log': tf.constant(1), 'logdir': debug_logdir, }
                )
        else:
            if lr_range_test:
                save_summaries_secs = 10
                save_interval_secs = 10000  # effectively deactivates saving checkpoints when doing LR range tests
            else:
                save_summaries_secs = 180
                save_interval_secs = 300  # Save one checkpoint once every 5 minutes

            if checkpoints is not None:
                if valid_iters > 0:
                    final_loss = slim.learning.train(
                        training_op,
                        log_dir,
                        # local_init_op=local_init_op,
                        # session_config=tf.ConfigProto(allow_soft_placement=True),
                        global_step=checkpoint_global_step_tensor,
                        save_summaries_secs=save_summaries_secs,
                        save_interval_secs=save_interval_secs,
                        number_of_steps=training_schedule['max_iters'],
                        train_step_fn=train_step_fn,
                        saver=saver,
                        init_fn=init_fn,
                        # summary_writer=train_writer,

                    )
                else:
                    final_loss = slim.learning.train(
                        training_op,
                        log_dir,
                        # local_init_op=local_init_op,
                        # session_config=tf.ConfigProto(allow_soft_placement=True),
                        global_step=checkpoint_global_step_tensor,
                        save_summaries_secs=save_summaries_secs,
                        save_interval_secs=save_interval_secs,
                        number_of_steps=training_schedule['max_iters'],
                        saver=saver,
                        init_fn=init_fn,
                        # summary_writer=train_writer,
                    )
            else:
                if valid_iters > 0:
                    final_loss = slim.learning.train(
                        training_op,
                        log_dir,
                        # session_config=tf.ConfigProto(allow_soft_placement=True),
                        global_step=checkpoint_global_step_tensor,
                        save_summaries_secs=save_summaries_secs,
                        save_interval_secs=save_interval_secs,
                        number_of_steps=training_schedule['max_iters'],
                        train_step_fn=train_step_fn,
                        saver=saver,
                        # summary_writer=train_writer,
                    )
                else:
                    final_loss = slim.learning.train(
                        training_op,
                        log_dir,
                        # session_config=tf.ConfigProto(allow_soft_placement=True),
                        global_step=checkpoint_global_step_tensor,
                        save_summaries_secs=save_summaries_secs,
                        save_interval_secs=save_interval_secs,
                        number_of_steps=training_schedule['max_iters'],
                        # train_step_fn=train_step_fn,
                        saver=saver,
                        # summary_writer=train_writer,
                    )
            print("Finished training, last batch loss: {:^15.4f}".format(final_loss))
