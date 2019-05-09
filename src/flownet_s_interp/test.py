import argparse
import os
from ..net import Mode
from .flownet_s_interp import FlowNetS_interp

FLAGS = None


def main():
    # Create a new network
    net = FlowNetS_interp(mode=Mode.TEST)

    # Test on the data
    if os.path.isfile(FLAGS.input_a) and FLAGS.input_a[:-4] is not '.txt':  # pair of images (not a batch)
        net.test(
            checkpoint='./checkpoints/FlowNetS/flownet-S.ckpt-0',
            input_a_path=FLAGS.input_a,
            input_b_path=None,
            matches_a_path=FLAGS.matches_a,
            out_path=FLAGS.out,
            input_type=FLAGS.input_type,
            sparse_flow_path=FLAGS.sparse_flow
        )
    elif os.path.isdir(FLAGS.input_a) and FLAGS.input_a[:-4] is '.txt':  # txt with image list (batch-like)
        net.test_batch(
            checkpoint='./checkpoints/FlowNetS/flownet-S.ckpt-0',
            image_paths=FLAGS.image_paths,
            out_path=FLAGS.out,
            input_type=FLAGS.input_type,
        )
    else:
        raise ValueError("'input_a' is not valid, should be a path to a folder or a single image")


# TODO: compute default matching masks and sparse_flow for repo example
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_paths',
        type=str,
        required=True,
        help='Path to first image',
        default='data/samples/0img0.ppm'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Path to the output folder where the final flow (and/or) visualisation will be stored',
        default='./'
    )
    parser.add_argument(
        '--input_type',
        type=str,
        required=False,
        help='type of input (def: frame 1 + frame 2), alternative: frame 1 + matches'
    )
    FLAGS = parser.parse_args()

    # Verify arguments are valid
    if not os.path.exists(FLAGS.image_paths):
        raise ValueError('image_a path must exist')
    if not os.path.isdir(FLAGS.out):
        raise ValueError('out directory must exist')
    main()
