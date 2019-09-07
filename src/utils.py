import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
import argparse
import os
import numpy as np
from math import ceil


# TODO: include losses in new file losses.py if we use many
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# === Training policies/schedules ===
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
        'exponential': The learning rate varies between the minimum and maximum boundaries and each boundary value
        declines by an exponential factor of: gamma^global_step.
    Args:
        g_step_op: Session global step.
        base_lr: Initial learning rate and minimum bound of the cycle.
        max_lr:  Maximum learning rate bound.
        step_size: Number of iterations in half a cycle. The paper suggests 2-8 x training iterations in epoch.
        gamma: Constant in 'exponential' mode gamma**(global_step)
        mode: One of {'triangular', 'triangular2', 'exponential'}. Default 'triangular'.
        one_cycle: if true, follow the one cycle policy, annealing the LR at the end
        (see https://arxiv.org/abs/1803.09820)
        annealing_factor: if one_cycle, the annealing factor (e.g.: to what order of magnitude we reduce the base_lr in
        the annealing part at the end)
        op_name: String.  Optional name of the operation.
    Returns:
        The cyclic learning rate.
    """
    assert (mode in ['triangular', 'triangular2', 'exponential'])
    one_cycle_target = 2
    lr = tf.convert_to_tensor(base_lr)
    global_step = tf.cast(g_step_op, lr.dtype)
    step_size = tf.cast(step_size, lr.dtype)
    anneal_fact = tf.cast(annealing_factor, lr.dtype)
    one_cycle_tar = tf.cast(one_cycle_target, lr.dtype)
    one_cycle_value = tf.convert_to_tensor(one_cycle)

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
    # If one cycle, the final (current) LR is the one for the annealing phase: min_lr - min_lr * annealing_factor
    a2 = tf.cond(tf.logical_and(tf.equal(one_cycle_value, tf.constant(True)), tf.equal(cycle, one_cycle_tar)),
                 lambda: tf.subtract(lr, tf.multiply(lr, anneal_fact)), lambda: tf.subtract(max_lr, lr))

    clr = tf.multiply(a1, a2)

    if mode == 'triangular2' and not one_cycle:
        clr = tf.divide(clr, tf.cast(tf.pow(2, tf.cast(cycle - 1, tf.int32)), tf.float32))
    if mode == 'exponential' and not one_cycle:
        clr = tf.multiply(tf.pow(gamma, global_step), clr)

    new_lr = tf.cond(tf.logical_and(tf.equal(one_cycle_value, tf.constant(True)), tf.equal(cycle, one_cycle_tar)),
                     lambda: tf.subtract(lr, clr), lambda: tf.add(lr, clr))

    return new_lr


def _mom_cyclic(g_step_op, base_mom=None, max_mom=None, step_size=None, gamma=0.99994, mode='triangular',
                one_cycle=False, op_name=None):
    """Computes a cyclic momentum as in https://arxiv.org/abs/1803.09820. Notice we leave triangular2 and exponential ena-
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
        'exponential': The momentum varies between the minimum and maximum boundaries and each boundary value
        declines by an exponential factor of: gamma^global_step.
    Args:
        g_step_op: Session global step.
        base_mom: Initial momentum and minimum bound of the cycle.
        max_mom:  Maximum momentum bound.
        step_size: Number of iterations in half a cycle. The paper suggests 2-8 x training iterations in epoch (CLR).
        gamma: Constant in 'exponential' mode gamma**(global_step)
        mode: One of {'triangular', 'triangular2', 'exponential'}. Default 'triangular'.
        one_cycle: if true, follow the one cycle policy, keeping the momentum constant at the maximum bound
        op_name: String.  Optional name of the operation.
    Returns:
        The cyclic momentum.
    """
    assert (mode in ['triangular', 'triangular2', 'exponential'])
    one_cycle_target = 2
    mom = tf.convert_to_tensor(base_mom)
    global_step = tf.cast(g_step_op, mom.dtype)
    step_size = tf.cast(step_size, mom.dtype)
    one_cycle_tar = tf.cast(one_cycle_target, mom.dtype)
    one_cycle_value = tf.convert_to_tensor(one_cycle)

    # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
    double_step = tf.multiply(2., step_size)
    global_div_double_step = tf.divide(global_step, double_step)
    cycle = tf.floor(tf.add(1., global_div_double_step))

    def momentum_clr():
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
        if mode == 'exponential' and not one_cycle:
            cmom = tf.multiply(tf.pow(gamma, global_step), cmom)

        return tf.subtract(max_mom, cmom, name=op_name)
    # Momentum is kept constant at its maximum if we use one_cycle and we are on the annealing phase
    return_mom = tf.cond(tf.logical_and(tf.equal(one_cycle_value, tf.constant(True)), tf.equal(cycle, one_cycle_tar)),
                         lambda: tf.identity(max_mom, name=op_name), lambda: momentum_clr())

    return return_mom


# === Network losses ===
# TODO: the function below extend this function's functionality ==> can be removed
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
        return tf.reduce_sum(epe) / num_samples


def average_endpoint_error_hfem(labels, predictions, add_hfem='', lambda_w=2., perc_hfem=50, edges=None):
    """
    Given labels and predictions of size (N, H, W, 2), calculates average endpoint error:
    sqrt[sum_across_channels{(X - Y)^2}].
    Optionally adds Hard Flow Example Mining for the p percentage of pixels with largest error
    This resembles the approach in the paper "Deep Flow-Guided Video Inpainting" by Rui Xu et. al where they increase
    the contribution to the loss for what they consider hard flow examples (which happen to be mostly on edges).
    Originally they use the L1 norm.
    :param labels: ground truth flow predictions at a particular scale
    :param predictions: network outputted flow at a given scale
    :param add_hfem: whether to add HFEM loss or not ('edges' use edges, 'hard' use hardcases, '' for standard AEPE)
    :param lambda_w: weight of this Hard Flow Example Mining error (relative to AEPE)
    :param perc_hfem: percentage of pixels to consider as Hard Examples (i.e.: top 50%). Used only for 'hard'
    :param edges: edges computed by SED algorithm (not binary but smooth in [0,1])
    According to the config: https://github.com/nbei/Deep-Flow-Guided-Video-Inpainting/tree/master/tools
    Lambda_w = 2 in training from scratch, 1 in later fine-tuning (for 'hard' only, for 'edges', experiment a bit)
    In the original approach, they did not use edges because they were doing inpainting (in this case we try it too)

    We base the multi-scale loss formulation on Clement Pinard PyTorch re-implementation which was based off original
    source: https://github.com/ClementPinard/FlowNetPytorch/issues/17
    Notice that NVIDIA FlowNet2 implementation computes the mean of EPE maps so the weighting is inversely proportional
    Like Sampepose's and Pychard's: https://github.com/ClementPinard/FlowNetPytorch/blob/master/multiscaleloss.py#L16,
     NVIDIA's: https://github.com/NVIDIA/flownet2-pytorch/blob/master/losses.py#L12
    """
    # 0. Convert all variables that are not tensors into tensors
    lambda_w = tf.convert_to_tensor(lambda_w, name='lambda')
    perc_hfem = tf.convert_to_tensor(perc_hfem, name='perc_hardflows')
    batch_size = predictions.shape.as_list()[0]

    # 1. Compute "standard AEPE"
    predictions = tf.to_float(predictions)
    labels = tf.to_float(labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    # aepe = ||labels-predictions||2 = 1/N sum(sqrt((labels - predictions)**2)
    # double-checked formulas thanks to: https://github.com/ppliuboy/SelFlow
    epe = tf.norm(tf.subtract(labels, predictions), axis=3, keepdims=True)  # defaults to euclidean norm
    epe_sum = tf.reduce_sum(epe)
    # loss_mean = epe_sum / tf.cast(tf.reduce_prod(tf.shape(predictions)[:-1]), tf.float32)
    aepe = tf.divide(epe_sum, batch_size)

    # 2. Compute HFEM loss
    if add_hfem:  # add_hfem not empty
        if add_hfem.lower() == 'hard':
            # 2.0.1 Average epe error 'images' over all batch (so we penalise top-k pixels w. largest MEAN error)
            epe_average_batch = epe  # tf.reduce_mean(epe, 0, keep_dims=True)
            # 2.0.2 Flatten EPE matrix to make finding idxs etc. easier
            epe_flatten = tf.reshape(epe_average_batch, [-1])   # Reshape (flatten) EPE

            # 2.1. Pick first p percentage (w. corresponding indices)
            # top_k = int(np.round((perc_hm / 100) * np.prod(epe.shape)))
            a1 = tf.divide(perc_hfem, 100)
            a2 = tf.cast(tf.reduce_prod(tf.shape(epe)), tf.float32)
            a1_times_a2 = tf.multiply(a1, a2)
            top_k = tf.cast(tf.round(a1_times_a2), tf.int32)  # compute the number of samples considered hard
            epe_top_k, epe_top_k_idxs = tf.nn.top_k(epe_flatten, k=top_k)  # get top_k largest elements in one go!

            # 2.2. Create mask to only take into account pixels in step 2
            HM_mask = tf.Variable(tf.zeros(tf.shape(epe_flatten)), trainable=False)
            ones = tf.Variable(tf.ones(tf.shape(epe_top_k_idxs)), trainable=False)
            HM_mask = tf.scatter_update(HM_mask, epe_top_k_idxs, ones)

            # 2.3. Create term for "hard" pixels, 0 everywhere but in top_k_idxs and then multiplied by lambda
            epe_hfem = tf.Variable(tf.zeros(tf.shape(epe_flatten)), trainable=False)
            lambda_epe = tf.multiply(lambda_w, epe_flatten)
            lambda_epe_filtered = tf.cast(tf.boolean_mask(lambda_epe, HM_mask), dtype=tf.float32)
            epe_hfem = tf.scatter_update(epe_hfem, epe_top_k_idxs, lambda_epe_filtered)
            # 2.4. Add 'standard' AEPE and lambda*AEPE_hfem term
            epe_and_hfem = tf.add(epe_flatten, epe_hfem)  # lambda already included
            epe_and_hfem_sum = tf.reduce_sum(epe_and_hfem)
            # Divide by batch size like 'standard' aepe since we are summing instead of averaging
            aepe_with_hfem = tf.divide(epe_and_hfem_sum, batch_size)
            # 2.3. Compute AEPE for "hard" pixels (we only miss the tf.reduce_sum(loss)/ num_hard_examples
            # epe_flatten_filtered = epe_flatten[HM_mask == 1]
            # epe_flatten_filtered = tf.cast(tf.boolean_mask(epe_flatten, HM_mask), dtype=tf.float32)
            # aepe_hfem = lambda * sum(EPE[hard_flows]) / len(hard_flows)
            # sum_epe_flatten_filtered = tf.reduce_sum(epe_flatten_filtered)
            # Adding extra cost is like instead of multiplying epe * 1 we multiply it by (1+lambda) ONLY for hard pixels
            # epe_and_hfem = epe_flatten_filtered + lambda * epe_flatten_filtered = (1 + lambda) * epe_flatten_filtered
            # epe_and_hfem = tf.multiply(tf.add(1, lambda_w), epe_flatten_filtered)
            # num_hard_pixels = tf.cast(tf.size(epe_flatten_filtered), tf.float32)
            # epe_and_hfem_sum = tf.reduce_sum(epe_and_hfem)
            # aepe_with_hfem = tf.divide(epe_and_hfem_sum, num_hard_pixels)

            return aepe_with_hfem
        elif add_hfem.lower() == 'edges' and edges is not None:
            # Reshape into height x width (was batch x height x width x 1 to be fed to the network)
            # aepe_hfem_edges = lambda * edges_img * epe_img
            epe_times_edges = tf.multiply(epe, edges)  # both have shape: (batch, height, width, 1)
            # Sum epe weighted by the edges and 'standard' epe and THEN sum and divide over batch size
            epe_and_edges = tf.add(epe, tf.multiply(lambda_w, epe_times_edges))

            # Sum and divide by batch size
            epe_edges_sum = tf.reduce_sum(epe_and_edges)
            aepe_with_edges = tf.divide(epe_edges_sum, batch_size)
            return aepe_with_edges
        else:
            # Return Average EPE
            return aepe
    else:
        return aepe


def mean_endpoint_error(gt_flow, pred_flow):
    """
    Computes the mean average endpoint error (L2 norm) between a ground truth and a predicted flow
    :param gt_flow:
    :param pred_flow:
    :return:
    """
    EPE = tf.norm(gt_flow - pred_flow, axis=-1, keepdims=True)
    AEPE = tf.reduce_mean(EPE)
    return AEPE


# Losses from SelFlow, the top performing OF estimation method on Sintel (at the time of writing this)
# They do an unsupervised training + supervised fine-tuning only on Sintel (no need to use FC or FT3D!)
# The authors made their code available at: https://github.com/ppliuboy/SelFlow
# The article can be found at: https://arxiv.org/abs/1904.09117
# Currently (29/07/19), only the test code is available (training code + models to be released in August)
def robust_loss(diff, mask, c=3.0, alpha=1.):
    """
    *OWN* clarifications: alternative robust loss to the one described in the paper
    :param diff:
    :param mask:
    :param c:
    :param alpha:
    :return:
    """
    z_alpha = max(1, 2-alpha)
    diff =  z_alpha / alpha * (tf.pow( tf.square((diff)/c)/z_alpha + 1 , alpha/2.)-1)
    diff = tf.multiply(diff, mask)
    diff_sum = tf.reduce_sum(diff)
    loss_mean = diff_sum / (tf.reduce_sum(mask) * 2 + 1e-6)
    batch_size = tf.to_float(tf.shape(diff)[0])
    loss_sum = diff_sum / batch_size
    return loss_mean, loss_sum


def abs_robust_loss(diff, mask, q=0.4):
    """
    *OWN* clarifications: robust loss from the paper, defined as psi(x) = np.power((np.abs(x) + epsilon), q)
    loss_mean is the one returned in validation (loss_sum is ignored)
    in training...(complete when training code is available) but probably loss_sum is the one used (we need an scalar
    to assign a loss to a given batch to guide training)
    :param diff: input tensor representing the difference between two other tensors (one may be ground truth or not)
    :param mask: mask specifying the valid pixels where the robust loss is defined (e.g.: only occluded, only non-occ.)
    :param q: smoothing factor
    :return:
    """
    diff = tf.pow((tf.abs(diff)+0.01), q)
    diff = tf.multiply(diff, mask)
    diff_sum = tf.reduce_sum(diff)
    loss_mean = diff_sum / (tf.reduce_sum(mask) * 2 + 1e-6)
    batch_size = tf.to_float(tf.shape(diff)[0])
    loss_sum = diff_sum / batch_size
    return loss_mean, loss_sum


# === Custom network layers/operations ===
# Thanks, https://github.com/tensorflow/tensorflow/issues/4079
# Replaced by tensorflow built-in (should be quicker?)
def LeakyReLU(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1.0 + leak)
        f2 = 0.5 * (1.0 - leak)
        return f1 * x + f2 * abs(x)


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


# TODO: we use a cuda operator to do the warping ATM, if we want to use occlusion(), we would need to adapt it
# Auxiliar functions to warp an image according to a flowfield
def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def tf_warp(img, flow, H, W):
    #    H = 256
    #    W = 256
    x,y = tf.meshgrid(tf.range(W), tf.range(H))
    x = tf.expand_dims(x,0)
    x = tf.expand_dims(x,-1)

    y = tf.expand_dims(y,0)
    y = tf.expand_dims(y,-1)

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    grid = tf.concat([x,y],axis = -1)
#    print grid.shape
    flows = grid+flow
    #print(flows.shape)
    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)
    zero = tf.zeros([], dtype=tf.int32)

    x = flows[:,:,:, 0]
    y = flows[:,:,:, 1]
    x0 = x
    y0 = y
    x0 = tf.cast(x0, tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(y0,  tf.int32)
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)


    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    return out


# From SelFlow (FWD-BWD consistency check with a different thresholding condition)
def length_sq(x):
    return tf.reduce_sum(tf.square(x), 3, keepdims=True)


def occlusion(flow_fw, flow_bw):
    x_shape = tf.shape(flow_fw)
    H = x_shape[1]
    W = x_shape[2]
    flow_bw_warped = tf_warp(flow_bw, flow_fw, H, W)
    flow_fw_warped = tf_warp(flow_fw, flow_bw, H, W)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)
    mag_sq_bw = length_sq(flow_bw) + length_sq(flow_fw_warped)
    occ_thresh_fw =  0.01 * mag_sq_fw + 0.5
    occ_thresh_bw =  0.01 * mag_sq_bw + 0.5
    occ_fw = tf.cast(length_sq(flow_diff_fw) > occ_thresh_fw, tf.float32)
    occ_bw = tf.cast(length_sq(flow_diff_bw) > occ_thresh_bw, tf.float32)

    return occ_fw, occ_bw


# Variational refinement used in InterpoNet (from EpicFlow)
def calc_variational_inference_map(imgA_filename, imgB_filename, flo_filename, out_filename, dataset):
    """
    Run the post processing variation energy minimization.
    :param imgA_filename: filename of RGB image A of the image pair.
    :param imgB_filename: filename of RGB image B of the image pair.
    :param flo_filename: filename of flow map to set as initialization.
    :param out_filename: filename for the output flow map.
    :param dataset: sintel / kitti
    """
    shell_command = '{0} {1} {2} {3} {4} {5}'.format('./src/SrcVariational/variational_main', imgA_filename,
                                                     imgB_filename, flo_filename, out_filename, dataset)
    print("cmd: {}".format(shell_command))
    print("cwd: {}".format(os.getcwd()))
    # shell_command = './SrcVariational/variational_main ' + imgA_filename + ' ' + imgB_filename + ' ' + flo_filename + \
    #                 ' ' + out_filename + ' -' + dataset
    exit_code = os.system(shell_command)
