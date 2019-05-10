from ..dataloader import load_batch
from ..dataset_configs import FLYING_CHAIRS_DATASET_CONFIG
from ..training_schedules import LONG_SCHEDULE
from .flownet2 import FlowNet2
import argparse


def main():
    # Create a new network
    net = FlowNet2()

    # Load a batch of data
    input_a, input_b, matches_a, sparse_flow, flow = load_batch(FLYING_CHAIRS_DATASET_CONFIG, 'sample', net.global_step,
                                                                input_type=FLAGS.input_type)

    # Train on the data
    net.train(
        log_dir='./logs/flownet_2',
        training_schedule=LONG_SCHEDULE,
        input_a=input_a,
        input_b=input_b,
        out_flow=flow,
        # Load trained weights for CSS and SD parts of network
        checkpoints={
            './checkpoints/FlowNetCSS-ft-sd/flownet-CSS-ft-sd.ckpt-0': ('FlowNet2/FlowNetCSS', 'FlowNet2'),
            './checkpoints/FlowNetSD/flownet-SD.ckpt-0': ('FlowNet2/FlowNetSD', 'FlowNet2')
        }
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_type',
        type=str,
        required=False,
        help='Path to first image',
        default='image_pairs'
    )
    FLAGS = parser.parse_args()
    main()
