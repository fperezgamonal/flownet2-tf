import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
import argparse
import numpy as np


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
# Which may (most likely) be based off: https://github.com/mhmoodlan/cyclic-learning-rate
def _lr_cyclic(g_step_op, base_lr=None, max_lr=None, step_size=None, gamma=0.99994, mode='triangular2', one_cycle=False,
               annealing_factor=1e-3, op_name=None):
    """Computes a cyclic learning rate, based on L.N. Smith's "Cyclical learning rates for training neural networks."
    [https://arxiv.org/pdf/1506.01186.pdf]
    This method lets the learning rate cyclically vary between the minimum (base_lr) and the maximum (max_lr)
    achieving improved classification accuracy and often in fewer iterations.
    08/07/19: added one_cycle policy based on the Keras implementation from:
     https://www.kaggle.com/robotdreams/one-cycle-policy-with-keras
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
        g_step_op: Session global step.
        base_lr: Initial learning rate and minimum bound of the cycle.
        max_lr:  Maximum learning rate bound.
        step_size: Number of iterations in half a cycle. The paper suggests 2-8 x training iterations in epoch.
        gamma: Constant in 'exp_range' mode gamma**(global_step)
        mode: One of {'triangular', 'triangular2', 'exp_range'}. Default 'triangular'.
        one_cycle: if true, follow the one cycle policy, annealing the LR at the end
        (see https://arxiv.org/abs/1803.09820)
        annealing_factor: if one_cycle, the annealing factor (e.g.: to what order of magnitude we reduce the base_lr in
        the annealing part at the end)
        op_name: String.  Optional name of the operation.
    Returns:
        The cyclic learning rate.
    """
    assert (mode in ['triangular', 'triangular2', 'exp_range'])
    lr = tf.convert_to_tensor(base_lr)
    global_step = tf.cast(g_step_op, lr.dtype)
    step_size = tf.cast(step_size, lr.dtype)
    anneal_fact = tf.cast(annealing_factor, lr.dtype)

    # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
    double_step = tf.multiply(2., step_size)
    global_div_double_step = tf.divide(global_step, double_step)
    cycle = tf.floor(tf.add(1., global_div_double_step))

    # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
    double_cycle = tf.multiply(2., cycle)
    global_div_step = tf.divide(global_step, step_size)
    tmp = tf.subtract(global_div_step, double_cycle)
    x = tf.abs(tf.add(1., tmp))

    a1 = tf.maximum(0., tf.subtract(1., x))  # max(0, 1-x)
    if one_cycle and cycle == 2:
        # computing: clr = learning_rate - ( learning_rate – learning_rate * anneal_factor ) * max( 0, 1 - x )
        a2 = tf.subtract(lr, tf.multiply(lr, anneal_fact))
    else:  # for anything else than one cycle with LR (not momentum, exitted above)
        # computing: clr = learning_rate + ( max_lr – learning_rate ) * max( 0, 1 - x )
        a2 = tf.subtract(max_lr, lr)
    clr = tf.multiply(a1, a2)

    if mode == 'triangular2' and not one_cycle:
        clr = tf.divide(clr, tf.cast(tf.pow(2, tf.cast(cycle - 1, tf.int32)), tf.float32))
    if mode == 'exp_range' and not one_cycle:
        clr = tf.multiply(tf.pow(gamma, global_step), clr)

    if one_cycle and cycle == 2:
        return tf.subtract(lr, clr, name=op_name)
    else:
        return tf.add(lr, clr, name=op_name)


def _mom_cyclic(g_step_op, base_mom=None, max_mom=None, step_size=None, gamma=0.99994, mode='triangular',
                one_cycle=False, op_name=None):
    """Computes a cyclic momentum as in https://arxiv.org/abs/1803.09820. Notice we leave triangular2 and exp_range ena-
    bled but we do not know if this types of policies help with a cyclical momentum with CLR (not used in 1cycle)
    This code returns the cyclic momentum computed as:
    ```python
    cycle = floor( 1 + global_step / ( 2 * step_size ) )
    x = abs( global_step / step_size – 2 * cycle + 1 )
    cmom = max_mom - ( max_mom – base_mom ) * max( 0 , 1 - x )
    ```
    Policies:
        'triangular': Default, linearly decreasing the momentum then linearly increasing it at each cycle (reverse beha-
        viour to the CLR).
        'triangular2': The same as the triangular policy except the momentum difference is cut in half at the end
        of each cycle. This means the momentum difference drops after each cycle.
        'exp_range': The momentum varies between the minimum and maximum boundaries and each boundary value
        declines by an exponential factor of: gamma^global_step.
    Args:
        g_step_op: Session global step.
        base_mom: Initial momentum and minimum bound of the cycle.
        max_mom:  Maximum momentum bound.
        step_size: Number of iterations in half a cycle. The paper suggests 2-8 x training iterations in epoch (CLR).
        gamma: Constant in 'exp_range' mode gamma**(global_step)
        mode: One of {'triangular', 'triangular2', 'exp_range'}. Default 'triangular'.
        one_cycle: if true, follow the one cycle policy, keeping the momentum constant at the maximum bound
        op_name: String.  Optional name of the operation.
    Returns:
        The cyclic momentum.
    """
    assert (mode in ['triangular', 'triangular2', 'exp_range'])
    mom = tf.convert_to_tensor(base_mom)
    global_step = tf.cast(g_step_op, mom.dtype)
    step_size = tf.cast(step_size, mom.dtype)

    # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
    double_step = tf.multiply(2., step_size)
    global_div_double_step = tf.divide(global_step, double_step)
    cycle = tf.floor(tf.add(1., global_div_double_step))

    if one_cycle and cycle == 2:
        return tf.identity(max_mom, name=op_name)
    else:
        # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
        double_cycle = tf.multiply(2., cycle)
        global_div_step = tf.divide(global_step, step_size)
        tmp = tf.subtract(global_div_step, double_cycle)
        x = tf.abs(tf.add(1., tmp))

        # computing: cmom = max_mom - ( max_mom – mom ) * max( 0, 1 - x )
        a1 = tf.maximum(0., tf.subtract(1., x))
        a2 = tf.subtract(max_mom, mom)
        cmom = tf.multiply(a1, a2)

        # Not tested (Leslie does not mention whether CM is used with CLR w. more than 1 cycle or how it is decayed
        if mode == 'triangular2' and not one_cycle:
            cmom = tf.divide(cmom, tf.cast(tf.pow(2, tf.cast(cycle - 1, tf.int32)), tf.float32))
        if mode == 'exp_range' and not one_cycle:
            cmom = tf.multiply(tf.pow(gamma, global_step), cmom)

        return tf.subtract(max_mom, cmom, name=op_name)


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
        epe = tf.reduce_sum(squared_difference, 3, keepdims=True)  # sum across channels (u,v)
        epe = tf.sqrt(epe)
        return tf.reduce_sum(epe) / num_samples, epe  # reduced to one number, mean/average epe (sum/num_samples)


def aepe_hfem(epe, lambda_w=0.01, perc_hfem=40):
    """
    Given labels and predictions of size (N, H, W, 2), calculates average endpoint error:
    sqrt[sum_across_channels{(X - Y)^2}] but only for the p percentage of pixels with largest error
    This resembles the approach in the paper "Deep Flow-Guided Video Inpainting" by Rui Xu et. al where they increase
    the contribution to the loss for what they consider hard flow examples (which happen to be mostly on edges).
    Originally they use the L1 norm.
    :param epe:  average endpoint error (still in matrix form!) based on which we define the hard examples as those with
     larger error
    :param lambda_w: weight of this Hard Flow Example Mining error (relative to AEPE)
    :param perc_hfem: percentage of pixels to consider as Hard Examples (i.e.: top 40%)
    Commented: numpy logic, above it's TF equivalent (luckily TF 2.0 will default to eager mode...)
    """
    # Convert all variables that are not tensors into tensors
    lambda_w = tf.convert_to_tensor(lambda_w, name='lambda')
    perc_hfem = tf.convert_to_tensor(perc_hfem, name='perc_hardflows')

    # 1. Sort pixels in decreasing order based on their EPE (decreasing)
    # epe_flatten = np.reshape(epe, np.product(epe.shape))  # flatten matrix for easier sorting
    epe_flatten = tf.reshape(epe, [-1])  # [1, tf.reduce_prod(tf.shape(epe))])
    # epe_f_top_err = np.argsort(epe_flatten)[::-1]  # sort in decreasing value
    epe_f_top_err = tf.argsort(epe_flatten, direction='DESCENDING')
    # 2. Pick first p percentage (w. corresponding indices)
    # top_k = int(np.round((perc_hm / 100) * np.prod(epe.shape)))  # compute the number of samples considered hard
    a1 = tf.divide(perc_hfem, 100)
    a2 = tf.cast(tf.reduce_prod(tf.shape(epe)), tf.float64)
    a1_times_a2 = tf.multiply(a1, a2)
    top_k = tf.cast(tf.round(a1_times_a2), tf.int32)
    # 3. Create mask to only take into account pixels in step 2
    # HM_mask = np.zeros(epe_flatten.shape)
    HM_mask = tf.Variable(tf.zeros(tf.shape(epe_flatten)))
    list_top_k = tf.range(0, top_k, dtype=tf.int32)  # [0, 1, ..., top_k-1]

    epe_top_k = tf.gather(epe_f_top_err, list_top_k)
    ones = tf.Variable(tf.ones(tf.shape(epe_top_k)))
    # HM_mask[epe_f_top_err[:top_k]] = 1
    HM_mask = tf.scatter_update(HM_mask, epe_top_k, ones)
    # Additional reshaping back to img space (not needed)
    # HM_mask = np.reshape(HM_mask, epe.shape)
    # HM_mask = tf.reshape(HM_mask, tf.reduce_prod(tf.shape(epe)))
    # 4. Compute AEPE for pixels in 2 (we only miss the tf.reduce_sum(loss)/ num_hard_examples
    # epe_flatten_filtered = epe_flatten[HM_mask == 1]
    epe_flatten_filtered = tf.cast(tf.boolean_mask(epe_flatten, HM_mask), dtype=tf.float32)
    tf.print(epe_flatten_filtered)
    # lambda * sum(EPE[hard_flows]) / len(hard_flows)
    sum_epe_flatten_filtered = tf.reduce_sum(epe_flatten_filtered)
    b1 = tf.multiply(lambda_w, sum_epe_flatten_filtered)
    b2 = tf.cast(tf.size(epe_flatten_filtered), tf.float32)
    return tf.divide(b1, b2)


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
