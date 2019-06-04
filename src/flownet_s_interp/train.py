from ..dataloader import load_batch
from .flownet_s_interp import FlowNetS_interp
import argparse


# TODO: update traning scripts for all other architectures with latest changes
def main():
    # Create a new network
    net = FlowNetS_interp()
    if FLAGS.checkpoint is not None and FLAGS.checkpoint:  # the second checks if the string is NOT empty
        checkpoints = FLAGS.checkpoint  # we want to define it as a string (only one checkpoint to load)
    else:
        checkpoints = None  # double-check None

    if FLAGS.input_type == 'image_matches':
        print("Input_type: 'image_matches'")
        input_a, matches_a, sparse_flow, flow = load_batch(FLAGS.dataset_config, 'train', input_type=FLAGS.input_type)

        # Train on the data
        net.train(
            log_dir='./logs/flownet_s_interp/image_matches',
            training_schedule_str=FLAGS.training_schedule,
            input_a=input_a,
            matches_a=matches_a,
            sparse_flow=sparse_flow,
            out_flow=flow,
            input_type=FLAGS.input_type,
            checkpoints=checkpoints,
            log_verbosity=FLAGS.log_verbosity,
            log_tensorboard=FLAGS.log_tensorboard,
            lr_range_test=FLAGS.lr_range_test,
        )
    else:
        print("Input_type: 'image_pairs'")
        input_a, input_b, flow = load_batch(FLAGS.dataset_config, 'train', input_type=FLAGS.input_type)

        # Train on the data
        net.train(
            log_dir='./logs/flownet_s_interp/image_pairs',
            training_schedule_str=FLAGS.training_schedule,
            input_a=input_a,
            input_b=input_b,
            out_flow=flow,
            input_type=FLAGS.input_type,
            checkpoints=checkpoints,
            log_verbosity=FLAGS.log_verbosity,
            log_tensorboard=FLAGS.log_tensorboard,
            lr_range_test=FLAGS.lr_range_test,
        )


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
        '--lr_range_test',
        type=bool,
        required=False,
        help='Whether or not to do a learning rate range test (exponentially) to find a good lr range',
        default=False,
    )
    parser.add_argument(
        '--log_verbosity',
        type=int,
        required=False,
        help='integer that specifies tf.logging verbosity (if <=1, INFO msg, >1, DEBUG msg)',
        default=1,
    )
    parser.add_argument(
        '--log_tensorboard',
        type=bool,
        required=False,
        help='Whether to log to Tensorboard or not (only stdout)',
        default=True,
    )

    FLAGS = parser.parse_args()

    main()
