import argparse
import os
from ..net import Mode
from .flownet_s_interp import FlowNetS_interp

FLAGS = None


def main():
    # Create a new network
    net = FlowNetS_interp(mode=Mode.TEST)

    # Test on the data
    if os.path.isfile(FLAGS.input_a):  # pair of images (not a batch)
        net.test(
            checkpoint='./checkpoints/FlowNetS/flownet-S.ckpt-0',
            input_a_path=FLAGS.input_a,
            input_b_path=FLAGS.matches_a,
            out_path=FLAGS.out,
            input_type=FLAGS.input_type,
        )
    elif os.path.isdir(FLAGS.input_a):  # folder of images (batch-like)
        net.test_batch(
            checkpoint='./checkpoints/FlowNetS/flownet-S.ckpt-0',
            input_a_path=FLAGS.input_a,
            input_b_path=FLAGS.matches_a,
            out_path=FLAGS.out,
            input_type=FLAGS.input_type,
        )
    else:
        raise ValueError("'input_a' is not valid, should be a path to a folder or a single image")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_a',
        type=str,
        required=True,
        help='Path to first image'
    )
    parser.add_argument(
        '--matches_a',
        type=str,
        required=True,
        help='Path to first to second image matches'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Path to output flow result'
    )
    parser.add_argument(
        '--input_type',
        type=str,
        required=False,
        help='type of input (def: frame 1 + frame 2), alternative: frame 1 + matches'
    )
    FLAGS = parser.parse_args()

    # Verify arguments are valid
    if not os.path.exists(FLAGS.input_a):
        raise ValueError('image_a path must exist')
    if not os.path.exists(FLAGS.matches_a):
        raise ValueError('match_a path must exist')
    if not os.path.isdir(FLAGS.out):
        raise ValueError('out directory must exist')
    main()
