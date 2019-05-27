from ..dataloader import load_batch
from .flownet_s_interp import FlowNetS_interp
import argparse
import resource

# TODO: consider enabling training sequentially with several schedules with only one call to 'train'
# TODO: should load previous checkpoint to properly resume training
# TODO: sometimes we want the flexibility of deciding whether we want to use all finetuning steps or only a few


def main():
    # Create a new network
    net = FlowNetS_interp()
    if FLAGS.checkpoint is not None:
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

    FLAGS = parser.parse_args()
    main()
