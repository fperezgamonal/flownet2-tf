import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def exponentially_increasing_lr(global_step, min_lr=1e-10, max_lr=1, num_iters=10000, name=None):
    if global_step is None:
        raise ValueError("global_step is required for cyclic_learning_rate.")
    with ops.name_scope(name, "exp_incr_lr",
                        [min_lr, global_step]) as name:
        learning_rate = tf.convert_to_tensor(min_lr, name="learning_rate")
        global_step = tf.cast(global_step, learning_rate.dtype)

        def exp_incr_lr():
            """Helper to recompute learning rate; most helpful in eager-mode."""
            maxlr_div_minlr = tf.divide(max_lr, min_lr)
            power_iter = tf.divide(global_step, num_iters)
            pow_div = tf.pow(maxlr_div_minlr, power_iter)
            return tf.multiply(min_lr, pow_div, name=name)

        if not context.executing_eagerly():
            exp_incr_lr = exp_incr_lr()
        return exp_incr_lr


def exponentially_decreasing_lr(global_step, min_lr=1e-10, max_lr=1, num_iters=10000, name=None):
    if global_step is None:
        raise ValueError("global_step is required for cyclic_learning_rate.")
    with ops.name_scope(name, "exp_decr_lr",
                        [min_lr, global_step]) as name:
        learning_rate = tf.convert_to_tensor(min_lr, name="learning_rate")
        global_step = tf.cast(global_step, learning_rate.dtype)

        def exp_decr_lr():
            """Helper to recompute learning rate; most helpful in eager-mode."""
            minlr_div_maxlr = tf.divide(min_lr, max_lr)
            power_iter = tf.divide(global_step, num_iters)
            pow_div = tf.pow(minlr_div_maxlr, power_iter)
            return tf.multiply(max_lr, pow_div, name=name)

        if not context.executing_eagerly():
            exp_decr_lr = exp_decr_lr()
        return exp_decr_lr


# From https://github.com/philferriere/tfoptflow/blob/master/tfoptflow/lr.py
def _lr_cyclic(g_step_op, base_lr=None, max_lr=None, step_size=None, gamma=0.99994, mode='triangular2', op_name=None):
    """Computes a cyclic learning rate, based on L.N. Smith's "Cyclical learning rates for training neural networks."
    [https://arxiv.org/pdf/1506.01186.pdf]
    This method lets the learning rate cyclically vary between the minimum (base_lr) and the maximum (max_lr)
    achieving improved classification accuracy and often in fewer iterations.
    This code returns the cyclic learning rate computed as:
    ```python
    cycle = floor( 1 + global_step / ( 2 * step_size ) )
    x = abs( global_step / step_size – 2 * cycle + 1 )
    clr = learning_rate + ( max_lr – learning_rate ) * max( 0 , 1 - x )
    ```
    Policies:
        'triangular': Default, linearly increasing then linearly decreasing the learning rate at each cycle.
        'triangular2': The same as the triangular policy except the learning rate difference is cut in half at the end
        of each cycle. This means the learning rate difference drops after each cycle.
        'exp_range': The learning rate varies between the minimum and maximum boundaries and each boundary value
        declines by an exponential factor of: gamma^global_step.
    Args:
        global_step: Session global step.
        base_lr: Initial learning rate and minimum bound of the cycle.
        max_lr:  Maximum learning rate bound.
        step_size: Number of iterations in half a cycle. The paper suggests 2-8 x training iterations in epoch.
        gamma: Constant in 'exp_range' mode gamma**(global_step)
        mode: One of {'triangular', 'triangular2', 'exp_range'}. Default 'triangular'.
        name: String.  Optional name of the operation.  Defaults to 'CyclicLearningRate'.
    Returns:
        The cyclic learning rate.
    """
    assert (mode in ['triangular', 'triangular2', 'exp_range'])
    lr = tf.convert_to_tensor(base_lr)
    global_step = tf.cast(g_step_op, lr.dtype)
    step_size = tf.cast(step_size, lr.dtype)

    # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
    double_step = tf.multiply(2., step_size)
    global_div_double_step = tf.divide(global_step, double_step)
    cycle = tf.floor(tf.add(1., global_div_double_step))

    # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
    double_cycle = tf.multiply(2., cycle)
    global_div_step = tf.divide(global_step, step_size)
    tmp = tf.subtract(global_div_step, double_cycle)
    x = tf.abs(tf.add(1., tmp))

    # computing: clr = learning_rate + ( max_lr – learning_rate ) * max( 0, 1 - x )
    a1 = tf.maximum(0., tf.subtract(1., x))
    a2 = tf.subtract(max_lr, lr)
    clr = tf.multiply(a1, a2)

    if mode == 'triangular2':
        clr = tf.divide(clr, tf.cast(tf.pow(2, tf.cast(cycle - 1, tf.int32)), tf.float32))
    if mode == 'exp_range':
        clr = tf.multiply(tf.pow(gamma, global_step), clr)

    return tf.add(clr, lr, name=op_name)


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
