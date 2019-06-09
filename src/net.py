import abc
from enum import Enum
from math import ceil
import os
import tensorflow as tf
import numpy as np
import glob
import sys
import datetime
import uuid
from imageio import imread, imsave
from .flowlib import flow_to_image, write_flow, read_flow, compute_all_metrics, get_metrics
from .training_schedules import LONG_SCHEDULE, FINE_SCHEDULE, SHORT_SCHEDULE, FINETUNE_SINTEL_S1, FINETUNE_SINTEL_S2, \
    FINETUNE_SINTEL_S3, FINETUNE_SINTEL_S4, FINETUNE_SINTEL_S5, FINETUNE_KITTI_S1, FINETUNE_KITTI_S2,\
    FINETUNE_KITTI_S3, FINETUNE_KITTI_S4, FINETUNE_ROB, LR_RANGE_TEST, CLR_SCHEDULE
slim = tf.contrib.slim
from .cyclic_learning_rate import clr

VAL_INTERVAL = 1000  # each N samples, we evaluate the validation set


class Mode(Enum):
    TRAIN = 1
    TEST = 2


# def train_step_fn(sess, train_op, global_step, train_step_kwargs):
#     """
#     slim.learning.train_step():
#       train_step_kwargs = {summary_writer:, should_log:, should_stop:}
#
#     usage: slim.learning.train( train_op, logdir,
#                                 train_step_fn=train_step_fn,)
#     """
#     if hasattr(train_step_fn, 'step'):
#         train_step_fn.step += 1  # or use global_step.eval(session=sess)
#     else:
#         train_step_fn.step = global_step.eval(sess)
#
#     # calc training losses
#     total_loss, should_stop = slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)
#
#     # validate on interval
#     if train_step_fn.step % VAL_INTERVAL == 0:
#         validate_loss, validation_delta = sess.run([val_loss, summary_validation_delta])
#         print(">> global step {}:    train={}   validation={}  delta={}".format(train_step_fn.step,
#                                                                                 total_loss, validate_loss,
#                                                                                 validate_loss - total_loss))
#
#     return [total_loss, should_stop]


# TODO: it is likely that the optimizer state cannot be properly resumed since we do not pass saver to .train()
#   * In fact, probably ONLY the global step is properly recovered as we give it to create_train_op
#   * Kudos to https://github.com/tensorflow/tensorflow/issues/312#issuecomment-335039382 (the whole discussion):
#       * We can create a saver properly initialised with the checkpoint variables (allowing for change in name)
#       * Specify maximum checkpoints to keep, variable list and keep one each N hours
#       * Then this tf.Saver() can be passed to tf.slim.learning.train() and hopefully we can resume training:
#           * with a loss approximately of the same value of that yield before pausing/stopping training
def optimistic_restore_vars(model_checkpoint_path):
    print("model_checkpoint_path is {}".format(model_checkpoint_path))
    reader = tf.train.NewCheckpointReader(model_checkpoint_path)
    saved_shapes = reader.get_variable_to_shape_map()
    print("len(saved_shapes.keys()): {}".format(len(saved_shapes.keys())))
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    print("len(var_names): {}".format(len(var_names)))
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    return restore_vars


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

    # TODO: consider merging all 'sintel' and 'kitti' schedules into one as they are sequential stages (stop if needed)
    def get_training_schedule(self, training_schedule_str):
        if training_schedule_str.lower() == 'fine':  # normally applied after long on FlyingThings3D
            training_schedule = FINE_SCHEDULE
        elif training_schedule_str.lower() == 'short':  # quicker schedule to train on FlyingChairs (or any dataset)
            training_schedule = SHORT_SCHEDULE
        elif training_schedule_str.lower() == 'clr':  # cyclical learning rate
            training_schedule = CLR_SCHEDULE
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
            print("New scheme: first + matches (1st=> 2nd frame) given")  # only for debugging, remove afterwards
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
        :return: postprocessed flow (cropped to original size if need be)
        """
        if adapt_info is not None:  # must define it when padding!
            # pred_flows = pred_flows[:, 0:adapt_info[1], 0:adapt_info[2], :]  # batch!
            pred_flows = pred_flows[0:adapt_info[-3], 0:adapt_info[-2], :]  # one sample
        return pred_flows

    def postproc_y_hat_train(self, y_hat):
        """
        Postprocess the output flows during train mode
        :param y_hat: predictions
        :param adapt_info: None if input image is multiple of 'divisor', original size otherwise
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
            # print("Avoid 'double-defining' as None...")
        input_a, input_b, matches_a, sparse_flow, x_adapt_info = self.adapt_x(input_a, input_b,
                                                                              matches_a, sparse_flow)
        # if sparse_flow_path is not None and matches_a_path is not None and input_type == 'image_matches':
        #     input_a, input_b, matches_a, sparse_flow, x_adapt_info = self.adapt_x(input_a, input_b,
        #                                                                           matches_a, sparse_flow)
        # else:
        #     input_a, input_b, matches_a, sparse_flow, x_adapt_info = self.adapt_x(input_a, input_b)

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

        training_schedule = LONG_SCHEDULE
        predictions = self.model(inputs, training_schedule)
        pred_flow = predictions['flow']

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            # Read and process the resulting list, one element at a time
            with open(image_paths, 'r') as input_file:
                path_list = input_file.read()

            for img_idx in range(len(path_list)):
                # Read + pre-process files
                # Each line is split into a list with N elements (separator: blank space (" "))
                path_inputs = path_list[img_idx].split(' ')
                assert 2 <= len(path_inputs) <= 6, (
                    'More paths than expected. Expected: I1+I2 (2), I1+MM+SF(3), I1+MM+SF+GTF(4),'
                    '  I1+MM+SF+GT+OCC_MSK+INVMASK(5 to 6)')

                # TODO: check if putting the if statements that check the length under fewer external ifs is simpler
                # Something like:
                #   if(input_type=='image_pairs'):
                #       if compute_metrics:
                #           if len(...)
                #           (...)
                #       else:
                #           if len(...)
                #   elif input_....

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
                        basefile = image_paths.split()[-1]
                        logfile = basefile.replace('.txt', '_metrics.log')
                        with open(logfile, 'w') as logfile:
                            logfile.write(final_str_formated)
                    else:  # print to stdout
                        print(final_str_formated)

    def train(self, log_dir, training_schedule_str, input_a, out_flow, input_b=None, matches_a=None, sparse_flow=None,
              checkpoints=None, input_type='image_pairs', log_verbosity=1, log_tensorboard=True, lr_range_test=False,
              train_params_dict=None):
        # Add validation batches as input? Used only once every val_interval steps...?
        """
        runs training on the network from which this method is called.
        :param log_dir:
        :param training_schedule_str:
        :param input_a:
        :param out_flow:
        :param input_b:
        :param matches_a:
        :param sparse_flow:
        :param checkpoints:
        :param input_type:
        :param log_verbosity:
        :param log_tensorboard:
        :param lr_range_test:
        :param train_params_dict:

        :return:
        """
        if log_verbosity <= 1:  # print loss and tfinfo to stdout
            print("Logging messages from 'INFO' level or worse")
            tf.logging.set_verbosity(tf.logging.INFO)
        else:  # debug info (more verbose)
            print("Logging messages from 'DEBUG' level or worse (this is the most verbose)")
            tf.logging.set_verbosity(tf.logging.DEBUG)
            print("Logging to tensorboard: {}".format(log_tensorboard))

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

                init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, renamed_variables)
                # Initialise checkpoint for stacked nets with the global step as the number of the outermost net
                step_number = int(checkpoint_path.split('-')[-1])
                checkpoint_global_step_tensor = tf.Variable(step_number, trainable=False, name='global_step',
                                                            dtype='int64')
                # TODO: adapt resuming from saver to stacked architectures if it works for one standalone
                saver = None
            elif isinstance(checkpoints, str):
                print("checkpoints is str, {}".format(checkpoints))
                checkpoint_path = checkpoints
                variables_to_restore = slim.get_model_variables()
                if log_verbosity > 1:
                    print("Restoring the following variables from checkpoint (SLIM):")
                    for var in variables_to_restore:
                        print(var)
                    print("Finished printing list of restored variables")
                #
                # init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
                #     checkpoint_path, variables_to_restore)
                # # Initialise checkpoint by parsing checkpoint name (not ideal but reading from checkpoint variable is
                # # currently not working as expected
                # step_number = int(checkpoint_path.split('-')[-1])
                # checkpoint_global_step_tensor = tf.Variable(step_number, trainable=False, name='global_step',
                #                                             dtype='int64')

                # likely fix to not proper resuming (huge loss)
                step_number = int(checkpoint_path.split('-')[-1])
                checkpoint_global_step_tensor = tf.Variable(step_number, trainable=False, name='global_step',
                                                            dtype='int64')
                # path_to_checkpoint_fld = os.path.dirname(checkpoint_path)
                if log_verbosity > 1:
                    print("Path to checkpoint folder is: '{}'".format(os.path.dirname(checkpoint_path)))
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_path))

                if log_verbosity > 1:
                    print("Is ckpt None: {0}".format(ckpt is None))
                vars2restore = optimistic_restore_vars(ckpt.model_checkpoint_path)
                if log_verbosity > 1:
                    print("Listing variables that will be restored(optimistic_restore_vars), total: {}:".format(
                        len(vars2restore)))
                    for var in vars2restore:
                        print(var)

                saver = tf.train.Saver(
                    max_to_keep=3, keep_checkpoint_every_n_hours=2, var_list=vars2restore if checkpoint_path else None)

            else:
                raise ValueError("checkpoint should be a single path (string) or a dictionary for stacked networks")
        else:
            saver = None
            checkpoint_global_step_tensor = tf.Variable(0, trainable=False, name='global_step', dtype='int64')

        # Create an initial assignment function.
        def InitAssignFn(sess):
            sess.run(init_assign_op, init_feed_dict)

        training_schedule = self.get_training_schedule(training_schedule_str)
        if log_tensorboard:
            tf.summary.image("image_a", input_a, max_outputs=1)
            if matches_a is not None and sparse_flow is not None and input_type == 'image_matches':
                tf.summary.image("matches_a", matches_a, max_outputs=1)
                # Convert sparse flow to image-like (ONLY for visualization)
                # not padding needed ! (we do it as a pre-processing step when creating the tfrecord)
                # Sparse flow is very difficult to visualize (0 values are white) in TB (do not include it)
                # sparse_flow_0 = sparse_flow[0, :, :, :]
                # sparse_flow_0 = tf.py_func(flow_to_image, [sparse_flow_0], tf.uint8)
                # sparse_flow_1 = sparse_flow[1, :, :, :]
                # sparse_flow_1 = tf.py_func(flow_to_image, [sparse_flow_1], tf.uint8)
                # sparse_flow_img = tf.stack([sparse_flow_0, sparse_flow_1], 0)
                #
                # tf.summary.image("sparse_flow_img", sparse_flow_img, max_outputs=1)
            else:
                tf.summary.image("image_b", input_b, max_outputs=1)

        if lr_range_test:  # learning rate range test to bound max/min optimal learning rate (2015, Leslie N. Smith)
            if lr_range_test is not None:  # use the input params
                start_lr = train_params_dict['start_lr']
                decay_steps = train_params_dict['decay_steps']
                decay_rate = train_params_dict['decay_rate']  # > 1 so it exponentially increases, does not decay
            else:  # use default values
                start_lr = 1e-10
                decay_steps = 110
                decay_rate = 1.25

            learning_rate = tf.train.exponential_decay(
                start_lr, global_step=checkpoint_global_step_tensor,
                decay_steps=decay_steps, decay_rate=decay_rate)

        # maybe this "shield" of checking the fixed config is not needed (3 checks)
        elif isinstance(training_schedule['learning_rates'], str):  # we are using a non-piecewise learning
            # cyclical learning rate forked from: https://github.com/mhmoodlan/cyclic-learning-rate
            if training_schedule['learning_rates'].lower() == 'clr' and training_schedule_str == 'clr':
                learning_rate = clr.cyclic_learning_rate(
                    checkpoint_global_step_tensor, train_params_dict['clr_min_lr'], train_params_dict['clr_max_lr'],
                    train_params_dict['clr_stepsize'], train_params_dict['clr_gamma'], train_params_dict['clr_mode'],
                )
                # Change max_iter to correspond to the end of the last cycle (in fact is still approximate but closer)
                new_max_iters = 2 * train_params_dict['clr_stepsize'] * train_params_dict['clr_num_cycles']

                if log_verbosity > 1:
                    print("(CLR) Default max. number of iters being changed from {} to {}".format(
                        training_schedule['max_iters'], new_max_iters))
                training_schedule['max_iters'] = new_max_iters
            else:
                learning_rate = 3e-4  # for Adam only!

            # add other policies (1-cycle), cosine-decay, etc.
        else:

            learning_rate = tf.train.piecewise_constant(
                checkpoint_global_step_tensor,
                [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
                training_schedule['learning_rates'])

        # As suggested in https://arxiv.org/abs/1711.05101, weight decay is not properly implemented for Adam
        # They implement L2 regularization. The next expression applied to tf.train.AdamOptimizer effectively
        # decouples weight decay from weight update (decays weight BEFORE updating gradients)
        # It only has the desired effects for optimizers that DO NOT depend on the value of 'var' in the update step
        # TODO: try AdamW and see if it improves convergence (smaller loss and speed)
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            training_schedule['momentum'],
            training_schedule['momentum2'])

        # AdamW = tf.contrib.opt.extend_with_decoupled_weight_decay(optimizer)
        if log_tensorboard:
            # Add learning rate
            tf.summary.scalar('learning_rate', learning_rate)

        if matches_a is not None and sparse_flow is not None and input_type == 'image_matches':
            inputs = {
                'input_a': input_a,
                'matches_a': matches_a,
                'sparse_flow': sparse_flow,
            }
        else:
            inputs = {
                'input_a': input_a,
                'input_b': input_b,
            }
        predictions = self.model(inputs, training_schedule)
        total_loss = self.loss(out_flow, predictions)

        if log_tensorboard:
            tf.summary.scalar('loss', total_loss)  # otherwise only printed to stdout!
            # Show the generated flow in TensorBoard
            if 'flow' in predictions:
                pred_flow_0 = predictions['flow'][0, :, :, :]
                pred_flow_0 = tf.py_func(flow_to_image, [pred_flow_0], tf.uint8)
                # pred_flow_0 = tf.py_function(func=flow_to_image, inp=[pred_flow_0], Tout=tf.uint8)
                pred_flow_1 = predictions['flow'][1, :, :, :]
                pred_flow_1 = tf.py_func(flow_to_image, [pred_flow_1], tf.uint8)
                # pred_flow_1 = tf.py_function(func=flow_to_image, inp=[pred_flow_1], Tout=tf.uint8)
                pred_flow_img = tf.stack([pred_flow_0, pred_flow_1], 0)
                tf.summary.image('pred_flow', pred_flow_img, max_outputs=1)

            true_flow_0 = out_flow[0, :, :, :]
            true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
            # true_flow_0 = tf.py_function(func=flow_to_image, inp=[true_flow_0], Tout=tf.uint8)
            true_flow_1 = out_flow[1, :, :, :]
            true_flow_1 = tf.py_func(flow_to_image, [true_flow_1], tf.uint8)
            # true_flow_1 = tf.py_function(func=flow_to_image, inp=[true_flow_1], Tout=tf.uint8)
            true_flow_img = tf.stack([true_flow_0, true_flow_1], 0)
            tf.summary.image('true_flow', true_flow_img, max_outputs=1)

        # Create the train_op
        print("Creating training op...")
        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            summarize_gradients=False,
            global_step=checkpoint_global_step_tensor,
        )

        # Create unique logging dir to avoid overwritting of old data (e.g.: when comparing different runs)
        now = datetime.datetime.now()
        date_now = now.strftime('%d-%m-%y_%H-%M-%S')
        if 'sintel' in training_schedule_str.lower():  # put all Sintel fine-tuning under the same log folder
            training_schedule_fld = 'fine_sintel'
        elif 'fine' in training_schedule_str.lower():  # format it a little better, FT3D for FlyingThings3D
            training_schedule_fld = "Sfine_FT3D"
        elif 'clr' in training_schedule_str.lower():  # make it all caps for readability
            training_schedule_fld = "CLR"
        elif 'long' in training_schedule_str.lower():
            training_schedule_fld = "Slong_FC".format(training_schedule_str)  # FC for FlyingChairs
        else:
            training_schedule_fld = training_schedule_str  # leave it as is

        log_dir = os.path.join(log_dir, training_schedule_fld, date_now)

        print("Starting training...")
        if self.debug:
            debug_logdir = os.path.join(log_dir, 'debug')
            if not os.path.isdir(debug_logdir):
                os.makedirs(debug_logdir)
                print("debugging logdir is: {}".format(debug_logdir))
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess)
                slim.learning.train_step(
                    sess,
                    train_op,
                    checkpoint_global_step_tensor,
                    {
                        'should_trace': tf.constant(1),
                        'should_log': tf.constant(1),
                        'logdir': debug_logdir,
                    }
                )
        else:
            if lr_range_test:
                save_summaries_secs = 10
            else:
                save_summaries_secs = 180

            # If max_steps is passed as a parameter, it overrides max_iter which is configured in training_schedules.py
            if 'max_steps' in train_params_dict:
                training_schedule['max_iters'] = train_params_dict['max_steps']

            if checkpoints is not None:
                final_loss = slim.learning.train(
                    train_op,
                    log_dir,
                    # session_config=tf.ConfigProto(allow_soft_placement=True),
                    global_step=checkpoint_global_step_tensor,
                    save_summaries_secs=save_summaries_secs,
                    number_of_steps=training_schedule['max_iters'],
                    # init_fn=InitAssignFn,
                    # train_step_fn=train_step_fn,
                    saver=saver,
                )
            else:
                final_loss = slim.learning.train(
                    train_op,
                    log_dir,
                    # session_config=tf.ConfigProto(allow_soft_placement=True),
                    global_step=checkpoint_global_step_tensor,
                    save_summaries_secs=save_summaries_secs,
                    number_of_steps=training_schedule['max_iters'],
                    # train_step_fn=train_step_fn,
                    saver=saver,
                )
            print("Loss at the end of training is {}".format(final_loss))

    # TODO: manually inspect if ALL optimizer variables are properly resumed (loss explodes for a few iterations)
    # It is not clear if Adam can only save some variables or all
    #  See: https://www.tensorflow.org/alpha/guide/checkpoints#manually_inspecting_checkpoints
    # def finetuning(...)
