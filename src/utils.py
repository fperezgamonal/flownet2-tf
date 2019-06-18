import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context


def exponentially_increasing_lr(global_step, min_lr=1e-10, max_lr=1, num_iters=10000, name=None):
    if global_step is None:
        raise ValueError("global_step is required for cyclic_learning_rate.")
    with ops.name_scope(name, "exp_incr_lr",
                        [min_lr, global_step]) as name:
        learning_rate = ops.convert_to_tensor(min_lr, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)

        def exp_incr_lr():
            """Helper to recompute learning rate; most helpful in eager-mode."""
            maxlr_div_minlr = math_ops.divide(max_lr, min_lr)
            power_iter = math_ops.divide(global_step, num_iters)
            pow_div = math_ops.pow(maxlr_div_minlr, power_iter)
            return math_ops.multiply(min_lr, pow_div, name=name)

        if not context.executing_eagerly():
            exp_incr_lr = exp_incr_lr()
        return exp_incr_lr


# Thanks, https://github.com/tensorflow/tensorflow/issues/4079
# Replaced by tensorflow built-in (should be quicker?)
def LeakyReLU(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1.0 + leak)
        f2 = 0.5 * (1.0 - leak)
        return f1 * x + f2 * abs(x)


def average_endpoint_error(labels, predictions):
    """
    Given labels and predictions of size (N, H, W, 2), calculates average endpoint error:
        sqrt[sum_across_channels{(X - Y)^2}]
    """
    num_samples = predictions.shape.as_list()[0]
    with tf.name_scope(None, "average_endpoint_error", (predictions, labels)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())

        squared_difference = tf.square(tf.subtract(predictions, labels))
        # sum across channels: sum[(X - Y)^2] -> N, H, W, 1
        loss = tf.reduce_sum(squared_difference, 3, keepdims=True)
        loss = tf.sqrt(loss)
        return tf.reduce_sum(loss) / num_samples


def pad(tensor, num=1):
    """
    Pads the given tensor along the height and width dimensions with `num` 0s on each side
    """
    return tf.pad(tensor, [[0, 0], [num, num], [num, num], [0, 0]], "CONSTANT")


def antipad(tensor, num=1):
    """
    Performs a crop. "padding" for a deconvolutional layer (conv2d tranpose) removes
    padding from the output rather than adding it to the input.
    """
    batch, h, w, c = tensor.shape.as_list()
    return tf.slice(tensor, begin=[0, num, num, 0], size=[batch, h - 2 * num, w - 2 * num, c])
