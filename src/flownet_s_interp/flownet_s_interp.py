from ..net import Net, Mode
from ..utils import LeakyReLU, average_endpoint_error, pad, antipad, aepe_hfem
from ..downsample import downsample
import math
import tensorflow as tf
slim = tf.contrib.slim


# Modified FlowNetS: same architecture BUT different input: first image + matches location + sparse flow
class FlowNetS_interp(Net):

    def __init__(self, mode=Mode.TRAIN, debug=False):
        super(FlowNetS_interp, self).__init__(mode=mode, debug=debug)

    # TODO: see where we can output the OF confidence map (based off last conv layer => heat map like?)
    # TODO: "modernise" architecture (remove 7x7 and 5x5 filters by 3x3 and proper stride!)
    # TODO: take "hints" from PWC-Net as it is has more or less the same nº of weights
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
                    with slim.arg_scope([slim.conv2d_transpose], biases_initializer=tf.zeros_initializer()):
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
                    flow = tf.image.resize_bilinear(flow,
                                                    tf.stack([height, width]),
                                                    align_corners=True)

                    return {
                        'predict_flow6': predict_flow6,
                        'predict_flow5': predict_flow5,
                        'predict_flow4': predict_flow4,
                        'predict_flow3': predict_flow3,
                        'predict_flow2': predict_flow2,
                        'flow': flow,
                    }

    # TODO: we can add the losses before making the weighted addition since we weight the hard examples the same (scale)
    # TODO: make sure all losses are properly added (notice loss is not used but may be added "simply" by name?
    def loss(self, flow, predictions, add_hard_flow_mining=False, lambda_weight=0.1, hard_examples_perc=40):
        flow = flow * 0.05  # i.e.: flow / 20

        aepe_losses = []
        hfem_losses = []
        INPUT_HEIGHT, INPUT_WIDTH = float(flow.shape[1].value), float(flow.shape[2].value)

        # L2 loss between predict_flow6, blob23 (weighted w/ 0.32)
        predict_flow6 = predictions['predict_flow6']
        size = [predict_flow6.shape[1], predict_flow6.shape[2]]
        downsampled_flow6 = downsample(flow, size)
        scale6_aepe, scale6_epe = average_endpoint_error(downsampled_flow6, predict_flow6)
        aepe_losses.append(scale6_aepe)
        if add_hard_flow_mining:
            # Compute Hard Flow Example Mining
            scale6_aepe_hfem = aepe_hfem(epe=scale6_epe, lambda_w=lambda_weight, perc_hfem=hard_examples_perc)
            hfem_losses.append(scale6_aepe_hfem)

        # L2 loss between predict_flow5, blob28 (weighted w/ 0.08)
        predict_flow5 = predictions['predict_flow5']
        size = [predict_flow5.shape[1], predict_flow5.shape[2]]
        downsampled_flow5 = downsample(flow, size)
        scale5_aepe, scale5_epe = average_endpoint_error(downsampled_flow5, predict_flow5)
        aepe_losses.append(scale5_aepe)

        if add_hard_flow_mining:
            # Compute Hard Flow Example Mining
            scale5_aepe_hfem = aepe_hfem(epe=scale5_epe, lambda_w=lambda_weight, perc_hfem=hard_examples_perc)
            hfem_losses.append(scale5_aepe_hfem)

        # L2 loss between predict_flow4, blob33 (weighted w/ 0.02)
        predict_flow4 = predictions['predict_flow4']
        size = [predict_flow4.shape[1], predict_flow4.shape[2]]
        downsampled_flow4 = downsample(flow, size)
        scale4_aepe, scale4_epe = average_endpoint_error(downsampled_flow4, predict_flow4)
        aepe_losses.append(scale4_aepe)

        if add_hard_flow_mining:
            # Compute Hard Flow Example Mining
            scale4_aepe_hfem = aepe_hfem(epe=scale4_epe, lambda_w=lambda_weight, perc_hfem=hard_examples_perc)
            hfem_losses.append(scale4_aepe_hfem)

        # L2 loss between predict_flow3, blob38 (weighted w/ 0.01)
        predict_flow3 = predictions['predict_flow3']
        size = [predict_flow3.shape[1], predict_flow3.shape[2]]
        downsampled_flow3 = downsample(flow, size)
        scale3_aepe, scale3_epe = average_endpoint_error(downsampled_flow3, predict_flow3)
        aepe_losses.append(scale3_aepe)

        if add_hard_flow_mining:
            # Compute Hard Flow Example Mining
            scale3_aepe_hfem = aepe_hfem(epe=scale3_epe, lambda_w=lambda_weight, perc_hfem=hard_examples_perc)
            hfem_losses.append(scale3_aepe_hfem)

        # L2 loss between predict_flow2, blob43 (weighted w/ 0.005)
        predict_flow2 = predictions['predict_flow2']
        size = [predict_flow2.shape[1], predict_flow2.shape[2]]
        downsampled_flow2 = downsample(flow, size)
        scale2_aepe, scale2_epe = average_endpoint_error(downsampled_flow2, predict_flow2)
        aepe_losses.append(scale2_aepe)

        if add_hard_flow_mining:
            # Compute Hard Flow Example Mining
            scale2_aepe_hfem = aepe_hfem(epe=scale2_epe, lambda_w=lambda_weight, perc_hfem=hard_examples_perc)
            hfem_losses.append(scale2_aepe_hfem)

        # Add both losses (if it applies)
        if add_hard_flow_mining:
            total_multiscale_loss = aepe_losses + hfem_losses
        else:
            total_multiscale_loss = aepe_losses

        weighted_ms_loss = tf.losses.compute_weighted_loss(total_multiscale_loss, [0.32, 0.08, 0.02, 0.01, 0.005])
        # Make sure loss is present in the final collection:
        # Default collection is tf.GraphKeys.LOSSES (used for training, another one for validation)
        tf.losses.add_loss(weighted_ms_loss)
        # Return the 'total' loss: loss fns + regularization terms defined in the model
        return tf.losses.get_total_loss()
