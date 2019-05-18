from ..dataloader import load_batch
from ..dataset_configs import FLYING_CHAIRS_ALL_DATASET_CONFIG, FLYING_CHAIRS_ALL_DATASET_CONFIG
from ..training_schedules import LONG_SCHEDULE
from .flownet_s_interp import FlowNetS_interp
import argparse


def main():
    # Create a new network
    net = FlowNetS_interp()

    if FLAGS.input_type == 'image_matches':
        input_a, matches_a, sparse_flow, flow = load_batch(FLYING_CHAIRS_ALL_DATASET_CONFIG, 'train', net.global_step,
                                                           input_type=FLAGS.input_type)
        print("Input_type: 'image_matches'")

        # Train on the data
        net.train(
            log_dir='./logs/flownet_s_interp/image_matches',
            training_schedule=LONG_SCHEDULE,
            input_a=input_a,
            matches_a=matches_a,
            sparse_flow=sparse_flow,
            out_flow=flow,
            input_type=FLAGS.input_type,
        )
    else:
        input_a, input_b, flow = load_batch(FLYING_CHAIRS_ALL_DATASET_CONFIG, 'train', net.global_step,
                                            input_type=FLAGS.input_type)
        print("Input_type: 'image_pairs'")

        # Train on the data
        net.train(
            log_dir='./logs/flownet_s_interp/image_pairs',
            training_schedule=LONG_SCHEDULE,
            input_a=input_a,
            input_b=input_b,
            out_flow=flow,
            input_type=FLAGS.input_type,
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
    FLAGS = parser.parse_args()
    main()
