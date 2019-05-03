import abc
from enum import Enum
import os
import tensorflow as tf
from .flowlib import flow_to_image, write_flow
import numpy as np
from scipy.misc import imread, imsave
import uuid
from .training_schedules import LONG_SCHEDULE
slim = tf.contrib.slim
from math import ceil
import glob


class Mode(Enum):
    TRAIN = 1
    TEST = 2


class Net(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, mode=Mode.TRAIN, debug=False):
        self.global_step = slim.get_or_create_global_step()
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

    def adapt_x(self, input_a, input_b, divisor=64):
        """
        Adapts the input images/matches to the required network dimensions
        :param input_a: first image
        :param input_b: second image (or matches between first and second images)
        :param divisor: divisor by which all image sizes should be divisible by. In fact: divisor=2^num_pyramids, here 6
        :return: padded and normalised input_a and input_b
        """
        # Convert from RGB -> BGR
        # First + second image
        if input_b.shape[-1] == input_a.shape[-1]:
            print("Normal scheme: first + second frame given")  # only for debugging, remove afterwards
            input_b = input_b[..., [2, 1, 0]]
        else:
            print("New scheme: first + matches (1st=> 2nd frame) given")  # only for debugging, remove afterwards

        input_a = input_a[..., [2, 1, 0]]

        # Scale from [0, 255] -> [0.0, 1.0] if needed
        if input_a.max() > 1.0:
            input_a = input_a / 255.0
        if input_b.max() > 1.0:
            input_b = input_b / 255.0

        height_a, width_a, channels_a = input_a.shape
        height_b, width_b, channels_b = input_b.shape
        # Assert matching sizes
        assert (height_a == height_b and width_a == width_b)

        if not(input_b.shape[-1] == input_a.shape[-1]):
            assert (channels_b == 1)  # assert it is a valid mask (black/white)

        if height_a % divisor != 0 or width_a % divisor != 0:
            new_height = int(ceil(height_a / divisor) * divisor)
            new_width = int(ceil(width_a / divisor) * divisor)
            pad_height = new_height - height_a
            pad_width = new_width - width_a

            if self.mode == Mode.TRAIN:  # working with batches, adapt to match dimensions
                padding = []
            elif self.mode == Mode.TEST:  # working with pair of images (for now there is no inference on whole batches)
                # TODO: modify test.py so we can predict batches (much quicker for whole datasets) and simplify this!
                padding = [(0, pad_height), (0, pad_width), (0, 0)]

            input_a = np.pad(input_a, padding, mode='constant', constant_values=0.)
            input_b = np.pad(input_b, padding, mode='constant', constant_values=0.)

            return input_a, input_b

    def adapt_y(self, flow, divisor=64):
        """
        Adapts the ground truth flows to train to the required network dimensions
        :param flow: ground truth optical flow between first and second image
        :param divisor: divisor by which all image sizes should be divisible by. In fact: divisor=2^num_pyramids, here 6
        :return: padded ground truth optical flow
        """
        # Assert it is a valid flow
        assert(flow.shape[-1] == 2)
        height = flow.shape[-3]  # temporally as this to account for batch/no batch tensor dimensions
        width = flow.shape[-2]

        if height % divisor != 0 or width % divisor != 0:
            new_height = int(ceil(height / divisor) * divisor)
            new_width = int(ceil(width / divisor) * divisor)
            pad_height = new_height - height
            pad_width = new_width - width

            if self.mode == Mode.TRAIN:  # working with batches, adapt to match dimensions
                padding = []
            elif self.mode == Mode.TEST:  # working with pair of images (for now there is no inference on whole batches)
                # TODO: modify test.py so we can predict batches (much quicker for whole datasets) and simplify this!
                padding = [(0, pad_height), (0, pad_width), (0, 0)]

            flow = np.pad(flow, padding, mode='constant', constant_values=0.)

            return flow

    def postproc_y_hat_test(self, y_hat, adapt_info=None):
        """
        Postprocess the output flows during test mode
        :param y_hat: predictions
        :param adapt_info: None if input image is multiple of 'divisor', original size otherwise
        :return: postprocessed flow (cropped to original size if need be)
        """
        pred_flows = y_hat[0]
        if adapt_info is not None:  # must define it when padding!
            pred_flows = pred_flows[:, 0:adapt_info[1], 0:adapt_info[2], :]  # batch!

        return pred_flows

    def postproc_y_hat_train(self, y_hat):
        """
        Postprocess the output flows during train mode
        :param y_hat: predictions
        :param adapt_info: None if input image is multiple of 'divisor', original size otherwise
        :return: batch loss and metric
        """

        return y_hat[0], y_hat[1]

    # TODO: embed size adaption to new net function and include the changes to work in train (batch_size, channels,...)
    # based on github.com/philferriere/tfoptflow/blob/33e8a701e34c8ce061f17297d40619afbd459ade/tfoptflow/model_pwcnet.py
    # functions: adapt_x, adapt_y, postproc_y_hat (crop)
    def test(self, checkpoint, input_a_path, input_b_path, out_path, save_image=True, save_flo=True):
        input_a = imread(input_a_path)
        input_b = imread(input_b_path)

        input_a, input_b = self.adapt_x(input_a, input_b)

        # TODO: This is a hack, we should get rid of this
        # the author probably means that it should be chosen as an input parameter not hardcoded!
        training_schedule = LONG_SCHEDULE

        # TODO: to work "in batches", re-define inputs as placeholders and read in for loop from folder
        inputs = {
            'input_a': tf.expand_dims(tf.constant(input_a, dtype=tf.float32), 0),
            # leave it like that for mask? uint8 may cause mismatch format when concatenating (after normalisation we
            # probably have a float anyway (due to the .0 in 255.0))
            'input_b': tf.expand_dims(tf.constant(input_b, dtype=tf.float32), 0),
        }
        predictions = self.model(inputs, training_schedule)
        pred_flow = predictions['flow']

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            pred_flow = sess.run(pred_flow)[0, :, :, :]

            pred_flow = self.postproc_y_hat_test(pred_flow)

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
                full_out_path = os.path.join(out_path, unique_name + '.flo')
                write_flow(pred_flow, full_out_path)

    def test_batch(self, checkpoint, input_a_path, input_b_path, out_path, input_type='image_pair', save_image=True,
                   save_flo=True):
        # Build Graph
        input_a = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
        if input_type == 'image_pair':
            input_b = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
        elif input_type == 'image_matches':
            input_b = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 1])
        else:
            print("Invalid output type, reverting to default (pairs of RGB images)...")
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
            # Get image list
            img_types = ('*.png', '*.PNG', '*.jpg', '*.JPG', '*.ppm', '*.PPM')  # add any other necessary (??)
            img_list = []
            for files in img_types:
                img_list.extend(glob.glob(os.path.join(input_a_path, files)))
                img_list = sorted(img_list)  # very important to process them in order (not really for image+match)
            if img_list is None:
                raise ValueError('directory must be non-empty')

            for img_idx in range(len(img_list)):
                # Read + pre-processed files
                frame_0 = imread(img_list[img_idx])
                if input_type == 'image_pair':
                    frame_1 = imread(img_list[img_idx + 1])
                elif input_type == 'image_matches':
                    img_ext = img_list[img_idx][:-4]
                    frame_1 = imread(img_list[img_idx].replace(img_ext, '_mask'.join(img_ext)))
                else:
                    frame_1 = imread(img_list[img_idx + 1])

                frame_0, frame_1 = self.adapt_x(frame_0, frame_1)

                flow = sess.run(pred_flow, feed_dict={
                    input_a: frame_0,
                    input_b: frame_1,
                })[0, :, :, :]

                pred_flow = self.postproc_y_hat_test(flow)

                # unique_name = 'flow-' + str(uuid.uuid4())  completely random and not useful to evaluate metrics after!
                unique_name = img_list[img_idx].split('/')[-1][:-4]

                if save_image or save_flo:
                    if not os.path.isdir(out_path):
                        os.makedirs(out_path)

                if save_image:
                    flow_img = flow_to_image(pred_flow)
                    full_out_path = os.path.join(out_path, unique_name + '_viz.png')
                    imsave(full_out_path, flow_img)

                if save_flo:
                    full_out_path = os.path.join(out_path, unique_name + '.flo')
                    write_flow(pred_flow, full_out_path)

    def train(self, log_dir, training_schedule, input_a, input_b, flow, checkpoints=None):
        tf.summary.image("image_a", input_a, max_outputs=2)
        tf.summary.image("image_b", input_b, max_outputs=2)

        self.learning_rate = tf.train.piecewise_constant(
            self.global_step,
            [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
            training_schedule['learning_rates'])

        optimizer = tf.train.AdamOptimizer(
            self.learning_rate,
            training_schedule['momentum'],
            training_schedule['momentum2'])

        inputs = {
            'input_a': input_a,
            'input_b': input_b,
        }
        predictions = self.model(inputs, training_schedule)
        total_loss = self.loss(flow, predictions)
        tf.summary.scalar('loss', total_loss)

        if checkpoints:
            for (checkpoint_path, (scope, new_scope)) in checkpoints.iteritems():
                variables_to_restore = slim.get_variables(scope=scope)
                renamed_variables = {
                    var.op.name.split(new_scope + '/')[1]: var
                    for var in variables_to_restore
                }
                restorer = tf.train.Saver(renamed_variables)
                with tf.Session() as sess:
                    restorer.restore(sess, checkpoint_path)

        # Show the generated flow in TensorBoard
        if 'flow' in predictions:
            pred_flow_0 = predictions['flow'][0, :, :, :]
            pred_flow_0 = tf.py_func(flow_to_image, [pred_flow_0], tf.uint8)
            pred_flow_1 = predictions['flow'][1, :, :, :]
            pred_flow_1 = tf.py_func(flow_to_image, [pred_flow_1], tf.uint8)
            pred_flow_img = tf.stack([pred_flow_0, pred_flow_1], 0)
            tf.summary.image('pred_flow', pred_flow_img, max_outputs=2)

        true_flow_0 = flow[0, :, :, :]
        true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
        true_flow_1 = flow[1, :, :, :]
        true_flow_1 = tf.py_func(flow_to_image, [true_flow_1], tf.uint8)
        true_flow_img = tf.stack([true_flow_0, true_flow_1], 0)
        tf.summary.image('true_flow', true_flow_img, max_outputs=2)

        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            summarize_gradients=True)

        if self.debug:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess)
                slim.learning.train_step(
                    sess,
                    train_op,
                    self.global_step,
                    {
                        'should_trace': tf.constant(1),
                        'should_log': tf.constant(1),
                        'logdir': log_dir + '/debug',
                    }
                )
        else:
            slim.learning.train(
                train_op,
                log_dir,
                # session_config=tf.ConfigProto(allow_soft_placement=True),
                global_step=self.global_step,
                save_summaries_secs=60,
                number_of_steps=training_schedule['max_iter']
            )
