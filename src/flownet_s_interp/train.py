from ..dataloader import load_batch
from .flownet_s_interp import FlowNetS_interp
import argparse
from ..utils import str2bool


# TODO: update traning scripts for all other architectures with latest changes
def main():
    # Create a new network
    net = FlowNetS_interp(no_deconv_biases=FLAGS.no_deconv_biases)
    if FLAGS.checkpoint is not None and FLAGS.checkpoint:  # the second checks if the string is NOT empty
        checkpoints = FLAGS.checkpoint  # we want to define it as a string (only one checkpoint to load)
    else:
        checkpoints = None  # double-check None

    # initialise range test values (exponentially/linearly increasing lr to be tested)
    if FLAGS.lr_range_test and FLAGS.training_schedule.lower() == 'lr_range_test':
        train_params_dict = {
            'lr_range_mode': FLAGS.lr_range_mode,
            'start_lr': FLAGS.start_lr,
            'end_lr': FLAGS.end_lr,
            'lr_range_niters': FLAGS.lr_range_niters,
            'optimizer': FLAGS.optimizer,
            'momentum': FLAGS.momentum,
            'min_momentum': FLAGS.min_momentum,
            'max_momentum': FLAGS.max_momentum,
            'weight_decay': FLAGS.weight_decay,
            'clip_grad_norm': FLAGS.clip_grad_norm,
        }
    # Initialise CLR parameters (define dictionary).
    elif FLAGS.training_schedule.lower() == 'clr':
        train_params_dict = {
            'clr_min_lr': FLAGS.clr_min_lr,
            'clr_max_lr': FLAGS.clr_max_lr,
            'clr_stepsize': FLAGS.clr_stepsize,
            'clr_num_cycles': FLAGS.clr_num_cycles,
            'clr_gamma': FLAGS.clr_gamma,
            'clr_mode': FLAGS.clr_mode,
            'optimizer': FLAGS.optimizer,
            'momentum': FLAGS.momentum,
            'min_momentum': FLAGS.min_momentum,
            'max_momentum': FLAGS.max_momentum,
            'weight_decay': FLAGS.weight_decay,
            'clip_grad_norm': FLAGS.clip_grad_norm,
        }
    elif FLAGS.training_schedule.lower() == 'one_cycle':
        train_params_dict = {
            'clr_min_lr': FLAGS.clr_min_lr,
            'clr_max_lr': FLAGS.clr_max_lr,
            'clr_stepsize': FLAGS.clr_stepsize,
            'one_cycle_annealing_factor': FLAGS.one_cycle_annealing_factor,
            'optimizer': FLAGS.optimizer,
            'momentum': FLAGS.momentum,
            'min_momentum': FLAGS.min_momentum,
            'max_momentum': FLAGS.max_momentum,
            'weight_decay': FLAGS.weight_decay,
            'clip_grad_norm': FLAGS.clip_grad_norm,
        }
    # Initialise exponentially decreasing LR (CLR may be too aggressive for fine-tuning)
    elif FLAGS.training_schedule.lower() == 'exp_decr':
        train_params_dict = {
            'start_lr': FLAGS.start_lr,
            'end_lr': FLAGS.end_lr,
            'optimizer': FLAGS.optimizer,
            'momentum': FLAGS.momentum,
            'weight_decay': FLAGS.weight_decay,
            'clip_grad_norm': FLAGS.clip_grad_norm,
        }
    else:
        train_params_dict = None

    # Add max_steps if the user wishes to overwrite the config in training_schedules.py
    if FLAGS.max_steps is not None:
        train_params_dict['max_steps'] = FLAGS.max_steps

    if FLAGS.input_type == 'image_matches':
        print("Input_type: 'image_matches'")
        # Train
        input_a, matches_a, sparse_flow, flow = load_batch(
            FLAGS.dataset_config, 'train', input_type=FLAGS.input_type,
            common_queue_capacity=FLAGS.common_queue_capacity,
            common_queue_min=FLAGS.common_queue_min,
            num_threads=FLAGS.num_threads,
            capacity_in_batches_train=FLAGS.capacity_in_batches_train,
            capacity_in_batches_val=FLAGS.capacity_in_batches_val,
            batch_size=FLAGS.batch_size,)
        # Validation
        if FLAGS.val_iters > 0:
            val_input_a, val_matches_a, val_sparse_flow, val_flow = load_batch(
                FLAGS.dataset_config, 'valid',
                input_type=FLAGS.input_type,
                common_queue_capacity=FLAGS.common_queue_capacity,
                common_queue_min=FLAGS.common_queue_min,
                num_threads=FLAGS.num_threads,
                capacity_in_batches_train=FLAGS.capacity_in_batches_train,
                capacity_in_batches_val=FLAGS.capacity_in_batches_val,
                batch_size=FLAGS.batch_size,)

        else:
            val_input_a = None
            val_matches_a = None
            val_sparse_flow = None
            val_flow = None

        # Train on the data
        net.train(
            log_dir='./logs/flownet_s_interp/image_matches',
            training_schedule_str=FLAGS.training_schedule,
            input_a=input_a,
            matches_a=matches_a,
            sparse_flow=sparse_flow,
            gt_flow=flow,
            valid_iters=FLAGS.val_iters,
            val_input_a=val_input_a,
            val_matches_a=val_matches_a,
            val_sparse_flow=val_sparse_flow,
            val_gt_flow=val_flow,
            input_type=FLAGS.input_type,
            checkpoints=checkpoints,
            log_verbosity=FLAGS.log_verbosity,
            log_tensorboard=FLAGS.log_tensorboard,
            lr_range_test=FLAGS.lr_range_test,
            train_params_dict=train_params_dict,
            log_smoothed_loss=FLAGS.log_smoothed_loss,
            reset_global_step=FLAGS.reset_global_step,
            summarise_grads=FLAGS.summarise_grads,
            add_hfem=FLAGS.add_hard_flow_example_mining,
            lambda_w=FLAGS.hfem_lambda_w,
            hfem_perc=FLAGS.hfem_perc_hard,
            dataset_config_str=FLAGS.dataset_config,
        )
    else:
        print("Input_type: 'image_pairs'")
        # Train
        input_a, input_b, flow = load_batch(
            FLAGS.dataset_config, 'train', input_type=FLAGS.input_type,
            common_queue_capacity=FLAGS.common_queue_capacity,
            common_queue_min=FLAGS.common_queue_min,
            num_threads=FLAGS.num_threads,
            capacity_in_batches_train=FLAGS.capacity_in_batches_train,
            capacity_in_batches_val=FLAGS.capacity_in_batches_val,
            batch_size=FLAGS.batch_size,)
        # Validation
        if FLAGS.val_iters > 0:
            val_input_a, val_input_b,  val_flow = load_batch(
                FLAGS.dataset_config, 'valid', input_type=FLAGS.input_type,
                common_queue_capacity=FLAGS.common_queue_capacity,
                common_queue_min=FLAGS.common_queue_min,
                num_threads=FLAGS.num_threads,
                capacity_in_batches_train=FLAGS.capacity_in_batches_train,
                capacity_in_batches_val=FLAGS.capacity_in_batches_val,
                batch_size=FLAGS.batch_size,)
        else:
            val_input_a = None
            val_input_b = None
            val_flow = None

        # Train on the data
        net.train(
            log_dir='./logs/flownet_s_interp/image_pairs',
            training_schedule_str=FLAGS.training_schedule,
            input_a=input_a,
            input_b=input_b,
            gt_flow=flow,
            valid_iters=FLAGS.val_iters,
            val_input_a=val_input_a,
            val_input_b=val_input_b,
            val_gt_flow=val_flow,
            input_type=FLAGS.input_type,
            checkpoints=checkpoints,
            log_verbosity=FLAGS.log_verbosity,
            log_tensorboard=FLAGS.log_tensorboard,
            lr_range_test=FLAGS.lr_range_test,
            train_params_dict=train_params_dict,
            log_smoothed_loss=FLAGS.log_smoothed_loss,
            reset_global_step=FLAGS.reset_global_step,
            summarise_grads=FLAGS.summarise_grads,
            add_hfem=FLAGS.add_hard_flow_example_mining,
            lambda_w=FLAGS.hfem_lambda_w,
            hfem_perc=FLAGS.hfem_perc_hard,
            dataset_config_str=FLAGS.dataset_config,
        )


# TODO: think a better way to generate the dictionaries of training parameters (not fixed, step-wise policies)
# Instead of defining so many input arguments and then creating the dict.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_type',
        type=str,
        required=False,
        help='Type of input that is fed to the network',
        default='image_matches'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=False,
        help='Path to checkpoint to load and continue training',
        default=None
    )
    parser.add_argument(
        '--no_deconv_biases',
        type=str2bool,
        nargs='?',
        required=False,
        help="Whether or not to 'remove' the biases from the deconv. layers (OG code did it). By default ALL layers "
             "have biases",
        default=False,
    )
    parser.add_argument(
        '--dataset_config',
        type=str,
        required=False,
        help='Dataset configuration to be used in training (i.e.: the dataset, crop size and data transformations)',
        default='flying_chairs',
    )
    parser.add_argument(
        '--training_schedule',
        type=str,
        required=False,
        help='Training schedule (learning rate, weight decay, etc.)',
        default='long_schedule',
    )
    parser.add_argument(
        '--val_iters',
        type=int,
        required=False,
        help='Validation interval, that is, a validation step is performed every val_iters (< 0 to disable validation)',
        default=2000,
    )
    # ==== Learning rate range test parameters ====
    parser.add_argument(
        '--lr_range_test',
        type=str2bool,
        nargs='?',
        required=False,
        help='Whether or not to do a learning rate range test (exponentially) to find a good lr range',
        default=False,
    )
    parser.add_argument(
        '--lr_range_mode',
        type=str,
        required=False,
        help='Whether a linear or exponential range is used (the latter does not require max_lr/end_lr)',
        default='linear',  # linear uses CLR policy with stepsize=max_iters (that is, only the increasing cycle)
    )
    parser.add_argument(
        '--start_lr',
        type=float,
        required=False,
        help='Initial (min) learning rate for the range finder',
        default=1e-8,
    )
    parser.add_argument(
        '--end_lr',
        type=float,
        required=False,
        help='Ending (max) learning rate for the range finder',
        default=1e-2,
    )
    # Linear test
    # Must define max_iter = stepsize
    parser.add_argument(
        '--lr_range_niters',
        type=int,
        required=False,
        help='Number of iterations for the linear learning rate range test (def.=10000)',
        default=10000,
    )
    # ==== Cyclic Learning Rate (CLR) parameters ====
    parser.add_argument(
        '--clr_min_lr',
        type=float,
        required=False,
        help='Lower bound (minimum learning rate) of Cyclic Learning Rate policy',
        default=4.25e-6,  # flying chairs, find exploring exponential range from 1e-10 up to aprox. 1e-3 (diverged)
    )
    parser.add_argument(
        '--clr_max_lr',
        type=float,
        required=False,
        help='Upper bound (maximum learning rate) of Cyclic Learning Rate policy',
        default=1e-4,  # same as above
    )
    parser.add_argument(
        '--clr_stepsize',
        type=int,
        required=False,
        help='Length of half-cycle (recommended to be 2-10x training iters=#train_ex/batch_size)',
        default=11116,  # 4x for flying chairs (22232/8=2779, 2779*4=11116)
    )
    parser.add_argument(
        '--clr_num_cycles',
        type=int,
        required=False,
        help='Number of cycles to perform (defines max_iters as 2*stepsize*num_cycles)',
        default=4,
    )
    parser.add_argument(
        '--clr_gamma',
        type=float,
        required=False,
        help="Exponential weighting factor (close to 1) that decays the learning rate as iterations grow (only used "
             "if mode: 'exponential'",
        default=0.99994,
    )
    parser.add_argument(
        '--clr_mode',
        type=str,
        required=False,
        help="Type of cyclic learning rate: 'triangular', 'triangular2' or 'exponential' (see Leslie N.Smith paper "
             "for more information)",
        default=5,
    )
    parser.add_argument(
        '--one_cycle_annealing_factor',
        type=float,
        required=False,
        help="Annealing factor if 1-cycle policy is used (LR will decrease linearly between clr_min_lr and"
             " clr_min_lr * annealing_factor until max_iters is reached)",
        default=1e-3,
    )
    # Overrides value defined in selected schedule training_schedules.py
    parser.add_argument(
        '--max_steps',
        type=int,
        required=False,
        help="Maximum number of iterations to train for (Careful, this overrides the setting of the current training "
             "schedule defined in training_schedules.py)",
        default=None,
    )
    # Overrides Adam optimizer as the default
    parser.add_argument(
        '--optimizer',
        type=str,
        required=False,
        help="Optimizer to use (def. 'adam'). Values: 'sgd', 'momentum' or 'adamw (w. proper weight decay)'",
        default=None,
    )
    # Overrides default value if Momentum optimizer (SGD+momentum) is used
    parser.add_argument(
        '--momentum',
        type=float,
        required=False,
        help="(constant) Value of momentum for the Momentum optimizer (SGD+momentum). For cyclic momentum, set "
             "'min_momentum' and 'max_momentum' instead and define the training schedule as 'clr'"
             "be used",
        default=None,
    )
    # Defines lower bound for the cyclic momentum (if CLR is in use)
    parser.add_argument(
        '--min_momentum',
        type=float,
        required=False,
        help="minimum value of the momentum for the SGD + Momentum optimizer (cyclic momentum)",
        default=0.85,
    )
    # Defines lower bound for the cyclic momentum (if CLR is in use)
    parser.add_argument(
        '--max_momentum',
        type=float,
        required=False,
        help="maximum value of the momentum for the SGD + Momentum optimizer (cyclic momentum)",
        default=0.95,
    )
    # Actual weight decay for AdamW (or L2 regularisation for all other optimisers)
    parser.add_argument(
        '--weight_decay',
        type=float,
        required=False,
        help="Weight decay for AdamW (w. proper weight decay) or L2 regularisation for other optimisers",
        default=None,
    )

    # TODO: see why DEBUG mode cannot be activated (only INFO)
    # ==== Logging ====
    parser.add_argument(
        '--log_verbosity',
        type=int,
        required=False,
        help='integer that specifies tf.logging verbosity (if <=1, INFO msg, >1, DEBUG msg)',
        default=1,
    )
    parser.add_argument(
        '--log_tensorboard',
        type=str2bool,
        nargs='?',
        required=False,
        help='Whether to log to Tensorboard or not (only stdout)',
        default=True,
    )
    parser.add_argument(
        '--log_smoothed_loss',
        type=str2bool,
        nargs='?',
        required=False,
        help='Whether to log an additional smoothed version of the loss (running average)',
        default=False,
    )
    parser.add_argument(
        '--reset_global_step',
        type=str2bool,
        nargs='?',
        required=False,
        help='Whether to restart global step at 0 if a checkpoint is inputted (useful when resuming training for '
             'different dataset than the one where the network was trained)',
        default=False,
    )
    parser.add_argument(
        '--summarise_grads',
        type=str2bool,
        nargs='?',
        required=False,
        help='Whether to add weights distribution and histograms to Tensorboard or not (useful to debug))',
        default=False,
    )
    # ==== Gradient norm clipping (to mitigate exploding gradients) ====
    # Overrides training_schedules.py default
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        required=False,
        help='If > 0, clip gradient norms by this value (to mitigate exploding gradients)',
        default=0,
    )
    # ==== Batch size ====
    # Overrides training_schedules.py default
    parser.add_argument(
        '--batch_size',
        type=int,
        required=False,
        help='integer that specifies the batch size',
        default=None,
    )
    # ==== Losses =====
    # Add Hard Flow Example Mining params
    parser.add_argument(
        '--add_hard_flow_example_mining',
        type=str2bool,
        nargs='?',
        required=False,
        help='Whether to add an extra cost for the hard examples (defined as those with largest EPE loss)',
        default=False,
    )
    parser.add_argument(
        '--hfem_lambda_w',
        type=float,
        required=False,
        help='If we use HFEM, relative weight of the added loss (i.e.: AEPE + lambda * HFEM_AEPE)',
        default=2
    )
    parser.add_argument(
        '--hfem_perc_hard',
        type=float,
        required=False,
        help='If we use HFEM, percentage of pixels to be considered hard (i.e.: top 40%)',
        default=50.0,
    )
    # ==== Capacity of queues ====
    parser.add_argument(
        '--common_queue_capacity',
        type=int,
        required=False,
        help='integer that specifies the capacity of the queue when loading data from a slim.dataset',
        default=256,
    )
    parser.add_argument(
        '--common_queue_min',
        type=int,
        required=False,
        help='integer that specifies the minimum capacity of the queue when loading data from a slim.dataset',
        default=128,
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        required=False,
        help='number of threads to load data from a slim.dataset',
        default=8,
    )
    # ==== Capacity in batches (when feeding to train/val) ====
    parser.add_argument(
        '--capacity_in_batches_train',
        type=int,
        required=False,
        help='capacity in batches to feed to train (kept in memory?)',
        default=4,
    )
    parser.add_argument(
        '--capacity_in_batches_val',
        type=int,
        required=False,
        help='capacity in batches to feed to validation (kept in memory?)',
        default=1,
    )

    FLAGS = parser.parse_args()

    main()
