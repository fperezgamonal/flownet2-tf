from ..net import Net, Mode
from ..utils import LeakyReLU, average_endpoint_error_hfem, pad, antipad
from ..downsample import downsample
import math
import tensorflow as tf
slim = tf.contrib.slim


# Modified FlowNetS: same architecture BUT different input: first image + matches location + sparse flow
class FlowNetS_interp(Net):

    def __init__(self, mode=Mode.TRAIN, debug=False, no_deconv_biases=False):
        super(FlowNetS_interp, self).__init__(mode=mode, debug=debug)
        self.no_deconv_biases = no_deconv_biases

    # TODO: see where we can output the OF confidence map (based off last conv layer => heat map like?)
    # TODO: "modernise" architecture (remove 7x7 and 5x5 filters by 3x3 and proper stride!)
    # TODO: take "hints" from PWC-Net as it is has more or less the same nº of weights
    # TODO: migrate everything we can from tf.slim to tf 2.0-like sintaxis
    # Follow:
    def model(self, inputs, training_schedule, trainable=True):
        _, height, width, _ = inputs['input_a'].shape.as_list()
        stacked = False
        with tf.variable_scope('FlowNetS'):  # MUST NOT change the scope, otherwise the checkpoint won't load!
            # used only to create "stacked" CNNs! (FlowNetCSS etc)
            if 'warped' in inputs and 'flow' in inputs and 'brightness_error' in inputs:
                stacked = True
                concat_inputs = tf.concat([inputs['input_a'],
                                           inputs['input_b'],
                                           inputs['warped'],
                                           inputs['flow'],  # already normalised in previous net (FlowNetC in this case)
                                           inputs['brightness_error']], axis=3)
            else:
                # TODO: consider permutating order
                concat_inputs = tf.concat([inputs['input_a'],
                                           inputs['sparse_flow'] * 0.05,  # normalised as predicted flow (makes sense)
                                           inputs['matches_a']], axis=3)

            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                # Only backprop this network if trainable
                                trainable=trainable,
                                # He (aka MSRA) weight initialization
                                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                    factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32),
                                # LeakyReLU, changed custom to TF's built-in
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                # We will do our own padding to match the original Caffe code
                                padding='VALID',
                                reuse=tf.AUTO_REUSE):
                weights_regularizer = slim.l2_regularizer(training_schedule['l2_regularization'])
                with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                    with slim.arg_scope([slim.conv2d], stride=2):
                        # Must set reuse for the first convolution to None so we can take different-sizes images for
                        # validation and training (data augmentation) ==> several changes (add to TODO)
                        # Source: https://www.researchgate.net/post/In_tensorflow_how_to_make_a_two-stream_neural_network_share_the_same_weights_in_several_layers
                        conv_1 = slim.conv2d(pad(concat_inputs, 3), 64, 7, scope='conv1')
                        conv_2 = slim.conv2d(pad(conv_1, 2), 128, 5, scope='conv2')
                        conv_3 = slim.conv2d(pad(conv_2, 2), 256, 5, scope='conv3')

                    conv3_1 = slim.conv2d(pad(conv_3), 256, 3, scope='conv3_1')
                    with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3):
                        conv4 = slim.conv2d(pad(conv3_1), stride=2, scope='conv4')
                        conv4_1 = slim.conv2d(pad(conv4), scope='conv4_1')
                        conv5 = slim.conv2d(pad(conv4_1), stride=2, scope='conv5')
                        conv5_1 = slim.conv2d(pad(conv5), scope='conv5_1')
                    conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope='conv6')
                    conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope='conv6_1')

                    """ START: Refinement Network """
                    if self.no_deconv_biases:
                        # Do not define biases for deconvolutional layers (like in the og code)
                        biases_initializer = None
                    else:
                        # Add biases to deconv layers (zero initialised)
                        biases_initializer = tf.zeros_initializer()

                    with slim.arg_scope([slim.conv2d_transpose], biases_initializer=biases_initializer,):
                        predict_flow6 = slim.conv2d(pad(conv6_1), 2, 3,
                                                    scope='predict_flow6',
                                                    activation_fn=None)
                        deconv5 = antipad(slim.conv2d_transpose(conv6_1, 512, 4,
                                                                stride=2,
                                                                scope='deconv5'))
                        upsample_flow6to5 = antipad(slim.conv2d_transpose(predict_flow6, 2, 4,
                                                                          stride=2,
                                                                          scope='upsample_flow6to5',
                                                                          activation_fn=None))
                        concat5 = tf.concat([conv5_1, deconv5, upsample_flow6to5], axis=3)

                        predict_flow5 = slim.conv2d(pad(concat5), 2, 3,
                                                    scope='predict_flow5',
                                                    activation_fn=None)
                        deconv4 = antipad(slim.conv2d_transpose(concat5, 256, 4,
                                                                stride=2,
                                                                scope='deconv4'))
                        upsample_flow5to4 = antipad(slim.conv2d_transpose(predict_flow5, 2, 4,
                                                                          stride=2,
                                                                          scope='upsample_flow5to4',
                                                                          activation_fn=None))
                        concat4 = tf.concat([conv4_1, deconv4, upsample_flow5to4], axis=3)

                        predict_flow4 = slim.conv2d(pad(concat4), 2, 3,
                                                    scope='predict_flow4',
                                                    activation_fn=None)
                        deconv3 = antipad(slim.conv2d_transpose(concat4, 128, 4,
                                                                stride=2,
                                                                scope='deconv3'))
                        upsample_flow4to3 = antipad(slim.conv2d_transpose(predict_flow4, 2, 4,
                                                                          stride=2,
                                                                          scope='upsample_flow4to3',
                                                                          activation_fn=None))

                        concat3 = tf.concat([conv3_1, deconv3, upsample_flow4to3], axis=3)

                        predict_flow3 = slim.conv2d(pad(concat3), 2, 3,
                                                    scope='predict_flow3',
                                                    activation_fn=None)
                        deconv2 = antipad(slim.conv2d_transpose(concat3, 64, 4,
                                                                stride=2,
                                                                scope='deconv2'))
                        upsample_flow3to2 = antipad(slim.conv2d_transpose(predict_flow3, 2, 4,
                                                                          stride=2,
                                                                          scope='upsample_flow3to2',
                                                                          activation_fn=None))
                        concat2 = tf.concat([conv_2, deconv2, upsample_flow3to2], axis=3)

                        predict_flow2 = slim.conv2d(pad(concat2), 2, 3,
                                                    scope='predict_flow2',
                                                    activation_fn=None)
                    """ END: Refinement Network """

                    flow = predict_flow2 * 20.0
                    # TODO: Look at Accum (train) or Resample (deploy) to see if we need to do something different
                    # TODO: should use TF2.0 compat version as this has a bug
                    # bug: asymmetrical padding ==> see:
                    # https://stackoverflow.com/questions/50591669/tf-image-resize-bilinear-vs-cv2-resize/50611485#50611485
                    # Not too bad if we add the flag: half_pixel_centers=True
                    flow = tf.image.resize_bilinear(flow, tf.stack([height, width]), align_corners=True,)
                    # half_pixel_centers=True) ==> cluster version 1.12 does not have it (added on TF 1.13)

                    return {
                        'predict_flow6': predict_flow6,
                        'predict_flow5': predict_flow5,
                        'predict_flow4': predict_flow4,
                        'predict_flow3': predict_flow3,
                        'predict_flow2': predict_flow2,
                        'flow': flow,
                    }

    # Computes the AEPE or AEPE + HFEM loss for all model scales
    def loss(self, flow, predictions, add_hard_flow_mining='', lambda_weight=2., hard_examples_perc=50, edges=None):
        flow = flow * 0.05  # i.e.: flow / 20
        losses = []
        INPUT_HEIGHT, INPUT_WIDTH = float(flow.shape[1].value), float(flow.shape[2].value)
        predictions['flow'].get_shape().assert_is_compatible_with(flow.get_shape())

        # L2 loss between predict_flow6, blob23 (weighted w/ 0.32)
        predict_flow6 = predictions['predict_flow6']
        size = [predict_flow6.shape[1], predict_flow6.shape[2]]
        downsampled_flow6 = downsample(flow, size)
        if edges is not None and add_hard_flow_mining:
            # Must downsample edges image so we can compute weighted loss at each scale
            downsampled_edges6 = downsample(edges, size)
            downsampled_flow6.get_shape()[:-1].assert_is_compatible_with(downsampled_edges6.get_shape()[:-1])
        else:
            downsampled_edges6 = edges
        losses.append(average_endpoint_error_hfem(downsampled_flow6, predict_flow6, add_hfem=add_hard_flow_mining,
                                                  lambda_w=lambda_weight, perc_hfem=hard_examples_perc,
                                                  edges=downsampled_edges6,))

        # L2 loss between predict_flow5, blob28 (weighted w/ 0.08)
        predict_flow5 = predictions['predict_flow5']
        size = [predict_flow5.shape[1], predict_flow5.shape[2]]
        downsampled_flow5 = downsample(flow, size)
        if edges is not None and add_hard_flow_mining:
            # Must downsample edges image so we can compute weighted loss at each scale
            downsampled_edges5 = downsample(edges, size)
            downsampled_flow5.get_shape()[:-1].assert_is_compatible_with(downsampled_edges5.get_shape()[:-1])
        else:
            downsampled_edges5 = edges
        losses.append(average_endpoint_error_hfem(downsampled_flow5, predict_flow5, add_hfem=add_hard_flow_mining,
                                                  lambda_w=lambda_weight, perc_hfem=hard_examples_perc,
                                                  edges=downsampled_edges5))

        # L2 loss between predict_flow4, blob33 (weighted w/ 0.02)
        predict_flow4 = predictions['predict_flow4']
        size = [predict_flow4.shape[1], predict_flow4.shape[2]]
        downsampled_flow4 = downsample(flow, size)
        if edges is not None and add_hard_flow_mining:
            # Must downsample edges image so we can compute weighted loss at each scale
            downsampled_edges4 = downsample(edges, size)
            downsampled_flow4.get_shape()[:-1].assert_is_compatible_with(downsampled_edges4.get_shape()[:-1])
        else:
            downsampled_edges4 = edges
        losses.append(average_endpoint_error_hfem(downsampled_flow4, predict_flow4, add_hfem=add_hard_flow_mining,
                                                  lambda_w=lambda_weight, perc_hfem=hard_examples_perc,
                                                  edges=downsampled_edges4))

        # L2 loss between predict_flow3, blob38 (weighted w/ 0.01)
        predict_flow3 = predictions['predict_flow3']
        size = [predict_flow3.shape[1], predict_flow3.shape[2]]
        downsampled_flow3 = downsample(flow, size)
        if edges is not None and add_hard_flow_mining:
            # Must downsample edges image so we can compute weighted loss at each scale
            downsampled_edges3 = downsample(edges, size)
            downsampled_flow3.get_shape()[:-1].assert_is_compatible_with(downsampled_edges3.get_shape()[:-1])
        else:
            downsampled_edges3 = edges
        losses.append(average_endpoint_error_hfem(downsampled_flow3, predict_flow3, add_hfem=add_hard_flow_mining,
                                                  lambda_w=lambda_weight, perc_hfem=hard_examples_perc,
                                                  edges=downsampled_edges3))

        # L2 loss between predict_flow2, blob43 (weighted w/ 0.005)
        predict_flow2 = predictions['predict_flow2']
        size = [predict_flow2.shape[1], predict_flow2.shape[2]]
        downsampled_flow2 = downsample(flow, size)
        if edges is not None and add_hard_flow_mining:
            # Must downsample edges image so we can compute weighted loss at each scale
            downsampled_edges2 = downsample(edges, size)
            downsampled_flow2.get_shape()[:-1].assert_is_compatible_with(downsampled_edges2.get_shape()[:-1])
        else:
            downsampled_edges2 = edges
        losses.append(average_endpoint_error_hfem(downsampled_flow2, predict_flow2, add_hfem=add_hard_flow_mining,
                                                  lambda_w=lambda_weight, perc_hfem=hard_examples_perc,
                                                  edges=downsampled_edges2))

        loss = tf.losses.compute_weighted_loss(losses, [0.32, 0.08, 0.02, 0.01, 0.005])
        # Make sure loss is present in the final collection:
        # Default collection is tf.GraphKeys.LOSSES (used for training, another one for validation)
        tf.losses.add_loss(loss)  # without this it worked despite TF strongly recommending this for custom losses
        # Return the 'total' loss: loss fns + regularization terms defined in the model
        return tf.losses.get_total_loss()
