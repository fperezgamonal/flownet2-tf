import argparse
import os
from ..net import Mode
from .flownet_s_interp import FlowNetS_interp
from ..utils import str2bool
FLAGS = None
DEBUG = False


# TODO: update other architectures test.py and train.py scripts according to FlownetS + interp
def main():
    # Create a new network
    net = FlowNetS_interp(mode=Mode.TEST, no_deconv_biases=FLAGS.no_deconv_biases)

    if DEBUG:
        print("Input file extension (input_a) is: '{}'".format(FLAGS.input_a[-4:]))
        print("{}".format(FLAGS.input_a[-4:] == '.txt'))

    # If there is any compulsory entry missing, use default example
    if FLAGS.input_a is None or not FLAGS.input_a:
        FLAGS.input_a = 'data/samples/sintel/frame_00186.png'
        FLAGS.input_b = 'data/samples/sintel/frame_00187.png'
        FLAGS.matches_a = 'data/samples/sintel/frame_00186_dm_mask.png'
        FLAGS.sparse_flow = 'data/samples/sintel/frame_00186_dm_sparse_flow.flo'
        FLAGS.gt_flow = 'data/samples/sintel/frame_00186.flo'
        FLAGS.occ_mask = 'data/samples/sintel/frame_00186_occ_mask.png'
        FLAGS.inv_mask = 'data/samples/sintel/frame_00186_inv_mask.png'
    # Test on the data
    if os.path.isfile(FLAGS.input_a) and FLAGS.input_a[-4:] != '.txt':  # pair of images (not a batch)
        print("Inferring on 'single' mode...")
        net.test(
            checkpoint=FLAGS.checkpoint,
            input_a_path=FLAGS.input_a,
            input_b_path=FLAGS.input_b,
            matches_a_path=FLAGS.matches_a,
            out_path=FLAGS.out,
            input_type=FLAGS.input_type,
            sparse_flow_path=FLAGS.sparse_flow,
            gt_flow=FLAGS.gt_flow,
            save_flo=FLAGS.save_flo,
            save_image=FLAGS.save_image,
            compute_metrics=FLAGS.compute_metrics,
            occ_mask=FLAGS.occ_mask,
            inv_mask=FLAGS.inv_mask,
            variational_refinement=FLAGS.variational_refinement,
            log_metrics2file=FLAGS.log_metrics2file,
            new_par_folder=FLAGS.new_par_folder,
        )
    elif os.path.isfile(FLAGS.input_a) and FLAGS.input_a[-4:] == '.txt':  # txt with image list (batch-like)
        print("Inferring on 'batch' mode...")
        net.test_batch(
            checkpoint=FLAGS.checkpoint,
            image_paths=FLAGS.input_a,
            out_path=FLAGS.out,
            input_type=FLAGS.input_type,
            save_flo=FLAGS.save_flo,
            save_image=FLAGS.save_image,
            compute_metrics=FLAGS.compute_metrics,
            accumulate_metrics=FLAGS.accumulate_metrics,
            log_metrics2file=FLAGS.log_metrics2file,
            width=FLAGS.width,
            height=FLAGS.height,
            new_par_folder=FLAGS.new_par_folder,
            variational_refinement=FLAGS.variational_refinement,
        )
    else:
        raise ValueError("'input_a' is not valid, should be a path to a folder or a single image")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_a',
        type=str,
        required=False,
        help='Path to first image',
        default=None,
    )
    parser.add_argument(
        '--input_b',
        type=str,
        required=False,
        help='Path to second image (probably only used for variational refinement in this case...)',
        default=None,
    )
    parser.add_argument(
        '--matches_a',
        type=str,
        required=False,
        help='Path to matches mask',
        default=None,
    )
    parser.add_argument(
        '--sparse_flow',
        type=str,
        required=False,
        help='Sparse flow initialized from sparse matches',
        default=None,
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint to load for inference',
        default='./checkpoints/FlowNetS/flownet-S.ckpt-0'
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
        '--out',
        type=str,
        required=False,
        help='Path to the output folder where the final flow (and/or) visualisation will be stored',
        default='./'
    )
    parser.add_argument(
        '--input_type',
        type=str,
        required=False,
        help='type of input (def: frame 1 + frame 2), alternative: frame 1 + matches',
        default='image_matches'
    )
    parser.add_argument(
        '--save_image',
        type=str2bool,
        required=False,
        help='whether to save an colour-coded image of the predicted flow or not',
        default=True,
    )
    parser.add_argument(
        '--save_flo',
        type=str2bool,
        required=False,
        help='whether to save the raw predicted flow in .flo format (see Middlebury specification for more details)',
        default=True,
    )
    parser.add_argument(
        '--new_par_folder',
        type=str,
        required=False,
        help='for batch inference, instead of creating a subfolder with the first file parent name, we assign a custom',
        default=None
    )
    parser.add_argument(
        '--compute_metrics',
        type=str2bool,
        required=False,
        help='whether to compute error metrics or not (if True all available metrics are computed, check flowlib.py)',
        default=True,
    )
    parser.add_argument(
        '--accumulate_metrics',
        type=str2bool,
        required=False,
        help='for batch: whether to accumulate metrics to compute averages (excluding outliers: Inf, Nan) or not)',
        default=True,
    )
    parser.add_argument(
        '--log_metrics2file',
        type=str2bool,
        required=False,
        help='whether to log the metrics to a file instead of printing them to stdout',
        default=False,
    )
    parser.add_argument(
        '--gt_flow',
        type=str,
        required=False,
        help='Path to ground truth flow so we can compute error metrics',
        default=None,
    )
    parser.add_argument(
        '--occ_mask',
        type=str,
        required=False,
        help='Path to occlusions mask (1s indicate pixel is occluded, 0 otherwise)',
        default=None,
    )
    parser.add_argument(
        '--inv_mask',
        type=str,
        required=False,
        help='Path to invalid mask with pixels that should not be considered when computing metrics = 1(invalid flow)',
        default=None,
    )
    parser.add_argument(
        '--width',
        type=int,
        required=False,
        help='Optionally specify the image(s) width',
        default=1024
    )
    parser.add_argument(
        '--height',
        type=int,
        required=False,
        help='Optionally specify the image(s) height',
        default=436
    )
    parser.add_argument(
        '--variational_refinement',
        type=str2bool,
        nargs='?',
        required=False,
        help='Whether to perform the variational refinement of EpicFlow (InterpoNet too) to the output flow',
        default=False,
    )
    FLAGS = parser.parse_args()

    # Verify arguments are valid
    if not os.path.exists(FLAGS.input_a):
        raise ValueError('Path to input_a (first image) must exist')
    if DEBUG:
        print("Input path to file with several paths or to a single image is:\n{}".format(FLAGS.input_a))

    main()
