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

    # based on github.com/philferriere/tfoptflow/blob/33e8a701e34c8ce061f17297d40619afbd459ade/tfoptflow/model_pwcnet.py
    # functions: adapt_x, adapt_y, postproc_y_hat (crop)
    def adapt_x(self, input_a, input_b, sparse_flow=None, divisor=64):
        """
        Adapts the input images/matches to the required network dimensions
        :param input_a: first image
        :param input_b: second image (or matches between first and second images)
        :param sparse_flow: (optional) sparse optical flow field initialised from a set of sparse matches
        :param divisor: (optional) number by which all image sizes should be divisible by. divisor=2^n_pyram, here 6
        :return: padded and normalised input_a, input_b and sparse_flow (if needed)
        """
        # Convert from RGB -> BGR
        # First + second image
        if input_b.shape[-1] == input_a.shape[-1]:
            print("Normal scheme: first + second frame given")  # only for debugging, remove afterwards
            input_b = input_b[..., [2, 1, 0]]
        else:
            input_b = input_b[..., np.newaxis]  # from (height, width) to (height, width, 1)
            print("New scheme: first + matches (1st=> 2nd frame) given")  # only for debugging, remove afterwards

        input_a = input_a[..., [2, 1, 0]]

        # Scale from [0, 255] -> [0.0, 1.0] if needed
        if input_a.max() > 1.0:
            input_a = input_a / 255.0
        if input_b.max() > 1.0:
            input_b = input_b / 255.0

        height_a, width_a, channels_a = input_a.shape  # temporal hack so it works with any size (batch or not)
        height_b, width_b, channels_b = input_b.shape
        if not input_a.shape[-1] == input_b.shape[-1]:
            assert(channels_b == 1)  # valid (reshaped) B&W mask
        else:
            assert(channels_a == channels_b)  # images in the same colourspace!
        # Assert matching width and height
        assert (height_a == height_b and width_a == width_b)

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
            # if not input_a.shape[-1] == input_b.shape[-1]:
            input_b = np.pad(input_b, padding, mode='constant', constant_values=0.)

            if sparse_flow is not None:
                sparse_flow = np.pad(input_b, padding, mode='constant', constant_values=0.)
            # else:
            #    input_b = np.pad(input_b, padding, mode='constant', constant_values=0.)

            return input_a, input_b, sparse_flow, x_adapt_info

    def adapt_y(self, flow, divisor=64):
        """
        Adapts the ground truth flows to train to the required network dimensions
        :param flow: ground truth optical flow between first and second image
        :param divisor: divisor by which all image sizes should be divisible by. In fact: divisor=2^num_pyramids, here 6
        :return: padded ground truth optical flow
        """
        # Assert it is a valid flow
        assert(flow.shape[-1] == 2)
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

            return flow, y_adapt_info

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

    def test(self, checkpoint, input_a_path, input_b_path, out_path, input_type='image_pair', sparse_flow_path=None,
             save_image=True, save_flo=True):
        input_a = imread(input_a_path)
        input_b = imread(input_b_path)

        if sparse_flow_path is not None and input_type == 'image_matches':
            # Read sparse flow from file
            sparse_flow = flow_to_image(sparse_flow_path)
            assert (sparse_flow.shape[-1] == 2)  # assert it is a valid flow
        else:
            sparse_flow = None

        input_a, input_b, sparse_flow, x_adapt_info = self.adapt_x(input_a, input_b, sparse_flow)

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
                'input_b': tf.expand_dims(tf.constant(input_b, dtype=tf.float32), 0),
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
                full_out_path = os.path.join(out_path, unique_name + '.flo')
                write_flow(pred_flow, full_out_path)

    def test_batch(self, checkpoint, input_a_path, input_b_path, out_path, input_type='image_pair',
                   sparse_flow_path=None, save_image=True, save_flo=True):
        """
        Run inference on a set of images defined in .txt files
        :param checkpoint: the path to the pre-trained model weights
        :param input_a_path: path to the txt file where the image paths are listed
        :param input_b_path: path to the matches masks (location) for the images above
        :param out_path: output path where flows and visualizations should be stored
        :param input_type: whether we are dealing with two consecutive frames or one frame + matches (interpolation)
        :param sparse_flow_path: (optional) path to the txt file with the route to the initial sparse flow to densify
        :param save_image: whether to save a png visualization (Middlebury colour code) of the flow
        :param save_flo: whether to save the 'raw' .flo file (useful to compute errors and such)
        :return:
        """
        # Build Graph
        input_a = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
        if input_type == 'image_pair':
            input_b = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3])
        elif input_type == 'image_matches':
            input_b = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 1])
            sparse_flow = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 2])
        else:
            print("Invalid output type, reverting to default (pairs of RGB images)...")
            input_b = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3])

        if sparse_flow_path is None:
            inputs = {
                'input_a': input_a,
                'input_b': input_b,
            }
        else:
            inputs = {
                'input_a': input_a,
                'input_b': input_b,
                'sparse_flow': sparse_flow,
            }

        training_schedule = LONG_SCHEDULE
        predictions = self.model(inputs, training_schedule)
        pred_flow = predictions['flow']

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            # TODO: input two txt files with the filepaths instead (more robust to other supported image types, etc.)
            # Read all txt files, ensure they have matching length, sort them alphabetically to ensure matching info
            # Then, process the resulting lists, one element at a time
            # If they are not sorted naturally by a number, 'sorted' will mess it up!
            with open(input_a_path, 'r') as input_file:
                input_a_list = sorted(input_file.read())

            with open(input_b_path, 'r') as input_file:
                input_b_list = sorted(input_file.read())

            with open(sparse_flow_path, 'r') as input_file:
                sparse_flow_list = sorted(input_file.read())

            assert(len(input_a_list) == len(input_b_list) == len(sparse_flow_list),
                   'Fatal: the input text file had mismatched dimensions')

            # Get image list
            # img_types = ('*.png', '*.PNG', '*.jpg', '*.JPG', '*.ppm', '*.PPM')  # add any other necessary (??)
            # img_list = []
            # for files in img_types:
            #     img_list.extend(glob.glob(os.path.join(input_a_path, files)))
            #     img_list = sorted(img_list)  # very important to process them in order (not really for image+match)
            # if img_list is None:
            #     raise ValueError('directory must be non-empty')

            for img_idx in range(len(input_a_list)):
                # Read + pre-process files
                frame_0 = imread(input_a_list[img_idx])
                frame_1 = imread(input_a_list[img_idx])
                if input_type == 'image_matches':
                    sparse_flow = flow_to_image(sparse_flow_list[img_idx])
                else:
                    sparse_flow = None

                frame_0, frame_1, sparse_flow, x_adapt_info = self.adapt_x(frame_0, frame_1, sparse_flow)
                if sparse_flow is None:
                    flow = sess.run(pred_flow, feed_dict={
                        input_a: frame_0, input_b: frame_1,
                    })[0, :, :, :]
                else:
                    flow = sess.run(pred_flow, feed_dict={
                        input_a: frame_0, input_b: frame_1,
                        sparse_flow: sparse_flow,
                    })[0, :, :, :]

                if x_adapt_info is not None:
                    y_adapt_info = (x_adapt_info[-3], x_adapt_info[-2], 2)
                else:
                    y_adapt_info = None
                pred_flow = self.postproc_y_hat_test(flow, adapt_info=y_adapt_info)

                # unique_name = 'flow-' + str(uuid.uuid4())  completely random and not useful to evaluate metrics after!
                # TODO: modify to keep the folder structure (at least parent folder of the image) ==> test!
                parent_folder_name = input_a_list[img_idx].split('/')[-2]
                unique_name = input_a_list[img_idx].split('/')[-1][:-4]
                out_path = os.path.join(out_path, parent_folder_name)

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

    # TODO: add the option to resume training from checkpoint (saver) ==> fine-tuning
    # TODO: if we restore a checkpoint, can we use it to properly save the state with the correct global step?
    # Or we may overwrite it by mistake?! By now, use the default checkpoint
    def train(self, log_dir, training_schedule, input_a, input_b, out_flow, sparse_flow=None, checkpoints=None,
              new_ckpt=None):
        tf.summary.image("image_a", input_a, max_outputs=2)
        tf.summary.image("image_b", input_b, max_outputs=2)
        if sparse_flow is not None:
            tf.summary.image("sparse_flow", sparse_flow, max_outputs=2)

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
        total_loss = self.loss(out_flow, predictions)
        tf.summary.scalar('loss', total_loss)

        if checkpoints:
            for (checkpoint_path, (scope, new_scope)) in checkpoints.items():
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

        true_flow_0 = out_flow[0, :, :, :]
        true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
        true_flow_1 = out_flow[1, :, :, :]
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
