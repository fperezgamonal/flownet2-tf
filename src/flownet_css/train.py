from ..dataloader import load_batch
from ..dataset_configs import FLYING_CHAIRS_DATASET_CONFIG, FLYING_CHAIRS_ALL_DATASET_CONFIG
from ..training_schedules import LONG_SCHEDULE
from .flownet_css import FlowNetCSS
import argparse


def main():
    # Create a new network
    net = FlowNetCSS()

    if FLAGS.input_type == 'image_matches':
        input_a, matches_a, sparse_flow, flow = load_batch(FLAGS.dataset_config, 'train', net.global_step,
                                                           input_type=FLAGS.input_type)

        # Train on the data
        net.train(
            log_dir='./logs/flownet_css/image_matches',
            training_schedule_str=FLAGS.training_schedule,
            input_a=input_a,
            matches_a=matches_a,
            sparse_flow=sparse_flow,
            out_flow=flow,
            # Load trained weights for CS part of network
            checkpoints={
                './checkpoints/FlowNetCS/flownet-CS.ckpt-0': ('FlowNetCSS/FlowNetCS', 'FlowNetCSS')}
        )
    else:
        input_a, input_b, flow = load_batch(FLAGS.dataset_config, 'train', net.global_step,
                                            input_type=FLAGS.input_type)

        # Train on the data
        net.train(
            log_dir='./logs/flownet_css/image_pairs',
            training_schedule_str=FLAGS.training_schedule,
            input_a=input_a,
            input_b=input_b,
            out_flow=flow,
            # Load trained weights for CS part of network
            checkpoints={
                './checkpoints/FlowNetCS/flownet-CS.ckpt-0': ('FlowNetCSS/FlowNetCS', 'FlowNetCSS')}
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_type',
        type=str,
        required=False,
        help='Type of input that is fed to the network',
        default='image_pairs'
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
