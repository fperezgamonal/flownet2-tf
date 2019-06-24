import argparse
import os
import sys
import numpy as np
from progressbar import ProgressBar, Percentage, Bar
from imageio import imread
import tensorflow as tf
from math import ceil

FLAGS = None

# Values defined here: https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs
TRAIN = 1
VAL = 2
# Any other value means that some image is not used for a reason (e.g.: in FlyingThings3D, complicated examples)
DEBUG = False  # used to deal with "corrupted" TFrecords (see commit #607542f comments for related issues)


# https://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
def open_flo_file(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            # print("Reading {0} x {1} flo file".format(h, w))
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            return np.resize(data, (h, w, 2))


# https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/convert_to_records.py
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_example(image_a, image_b, matches_a, sparse_flow, flow):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'image_a': _bytes_feature(image_a),
        'image_b': _bytes_feature(image_b),
        'matches_a': _bytes_feature(matches_a),
        'sparse_flow': _bytes_feature(sparse_flow),
        'flow': _bytes_feature(flow),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto


def convert_dataset(indices, name, matcher='deepmatching', dataset='flying_chairs', divisor=64):
    # Open a TFRRecordWriter
    filename = os.path.join(FLAGS.out, name + '.tfrecords')
    writeOpts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    with tf.python_io.TFRecordWriter(filename, options=writeOpts) as writer:

        # Load each data sample (image_a, image_b, matches_a, sparse_flow, flow) and write it to the TFRecord
        count = 0
        pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(indices)).start()
        for i in indices:
            if dataset == 'flying_chairs':
                image_a_path = os.path.join(FLAGS.data_dir, '{0:05d}_img1.png'.format(i + 1))
                image_b_path = os.path.join(FLAGS.data_dir, '{0:05d}_img2.png'.format(i + 1))
                flow_path = os.path.join(FLAGS.data_dir, '{0:05d}_flow.flo'.format(i + 1))
                if matcher == 'sift':
                    matches_a_path = os.path.join(FLAGS.data_dir, '{0:05d}_img1_sift_mask.png'.format(i + 1))
                    sparse_flow_path = os.path.join(FLAGS.data_dir, '{0:05d}_img1_sift_sparse_flow.flo'.format(i + 1))
                elif matcher == 'deepmatching':
                    matches_a_path = os.path.join(FLAGS.data_dir, '{0:05d}_img1_dm_mask.png'.format(i + 1))
                    sparse_flow_path = os.path.join(FLAGS.data_dir, '{0:05d}_img1_dm_sparse_flow.flo'.format(i + 1))
                # add more matchers if need be (more elif's)
                else:
                    raise ValueError("Invalid matcher name. Available: ('deepmatching', 'sift')")
            elif dataset == 'flying_things3D':
                """
                    We have a stereo-pair but we will arbitrarily only use the 'left' view (any would be ok)
                    By doing so, we can only copy the left (forward) ground truth and significantly reduce the storage
                    burden.  Also, we follow FlowNet2.0 training and discard a set of difficult sequences (1388, 
                    to be precise)
                """
                image_a_path = os.path.join(FLAGS.data_dir, '{0:07d}.png'.format(i))
                image_b_path = os.path.join(FLAGS.data_dir, '{0:07d}.png'.format(i + 1))
                flow_path = os.path.join(FLAGS.data_dir, '{0:07d}.flo'.format(i))
                if matcher == 'sift':
                    matches_a_path = os.path.join(FLAGS.data_dir, '{0:07d}_sift_mask.png'.format(i))
                    sparse_flow_path = os.path.join(FLAGS.data_dir, '{0:07d}_sift_sparse_flow.flo'.format(i))
                elif matcher == 'deepmatching':
                    matches_a_path = os.path.join(FLAGS.data_dir, '{0:07d}_dm_mask.png'.format(i))
                    sparse_flow_path = os.path.join(FLAGS.data_dir, '{0:07d}_dm_sparse_flow.flo'.format(i))
                # add more matchers if need be (more elif's)
                else:
                    raise ValueError("Invalid matcher name. Available: ('deepmatching', 'sift')")

            elif dataset == 'sintel_clean':
                pass_dir = 'clean'
                image_a_path = os.path.join(FLAGS.data_dir, pass_dir, 'frame_{0:04d}.png'.format(i+1))
                image_b_path = os.path.join(FLAGS.data_dir, pass_dir, 'frame_{0:04d}.png'.format(i+2))
                flow_path = os.path.join(FLAGS.data_dir, pass_dir, 'frame_{0:04d}.flo'.format(i+1))
                if matcher == 'sift':
                    matches_a_path = os.path.join(FLAGS.data_dir, pass_dir, 'frame_{0:04d}_sift_mask.png'.format(i+1))
                    sparse_flow_path = os.path.join(FLAGS.data_dir, pass_dir,
                                                    'frame_{0:04d}_sift_sparse_flow.flo'.format(i+1))
                elif matcher == 'deepmatching':
                    matches_a_path = os.path.join(FLAGS.data_dir, pass_dir, 'frame_{0:04d}_dm_mask.png'.format(i+1))
                    sparse_flow_path = os.path.join(FLAGS.data_dir, pass_dir,
                                                    'frame_{0:04d}_dm_sparse_flow.flo'.format(i+1))
                else:
                    raise ValueError("Invalid matcher name. Available: ('deepmatching', 'sift')")

            elif dataset == 'sintel_final':
                pass_dir = 'final'
                image_a_path = os.path.join(FLAGS.data_dir, pass_dir, 'frame_{0:04d}.png'.format(i+1))
                image_b_path = os.path.join(FLAGS.data_dir, pass_dir, 'frame_{0:04d}.png'.format(i+2))
                flow_path = os.path.join(FLAGS.data_dir, pass_dir, 'frame_{0:04d}.flo'.format(i+1))
                if matcher == 'sift':
                    matches_a_path = os.path.join(FLAGS.data_dir, pass_dir, 'frame_{0:04d}_sift_mask.png'.format(i+1))
                    sparse_flow_path = os.path.join(FLAGS.data_dir, pass_dir,
                                                    'frame_{0:04d}_sift_sparse_flow.flo'.format(i+1))
                elif matcher == 'deepmatching':
                    matches_a_path = os.path.join(FLAGS.data_dir, pass_dir, 'frame_{0:04d}_dm_mask.png'.format(i+1))
                    sparse_flow_path = os.path.join(FLAGS.data_dir, pass_dir,
                                                    'frame_{0:04d}_dm_sparse_flow.flo'.format(i+1))
                else:
                    raise ValueError("Invalid matcher name. Available: ('deepmatching', 'sift')")
            elif dataset == 'sintel_all':
                pass_dir = 'final'
                image_a_path = os.path.join(FLAGS.data_dir, pass_dir, 'frame_{0:05d}.png'.format(i+1))
                image_b_path = os.path.join(FLAGS.data_dir, pass_dir, 'frame_{0:05d}.png'.format(i+2))
                flow_path = os.path.join(FLAGS.data_dir, pass_dir, 'frame_{0:05d}.flo'.format(i+1))
                if matcher == 'sift':
                    matches_a_path = os.path.join(FLAGS.data_dir, pass_dir, 'frame_{0:05d}_sift_mask.png'.format(i+1))
                    sparse_flow_path = os.path.join(FLAGS.data_dir, pass_dir,
                                                    'frame_{0:05d}_sift_sparse_flow.flo'.format(i+1))
                elif matcher == 'deepmatching':
                    matches_a_path = os.path.join(FLAGS.data_dir, pass_dir, 'frame_{0:05d}_dm_mask.png'.format(i+1))
                    sparse_flow_path = os.path.join(FLAGS.data_dir, pass_dir,
                                                    'frame_{0:05d}_dm_sparse_flow.flo'.format(i+1))
                else:
                    raise ValueError("Invalid matcher name. Available: ('deepmatching', 'sift')")
            # Add more datasets here
            # elif dataset == 'another_of_dataset':
            else:
                raise ValueError("Invalid dataset name!")

            if DEBUG:
                print("Path to source images/flows are:")
                print("img_a: {0}\nimg_b: {1}\nmch_a: {2}\nsp_flow: {3}\nflow: {4}\n".format(
                    image_a_path, image_b_path, matches_a_path, sparse_flow_path, flow_path))

            image_a = imread(image_a_path)
            image_b = imread(image_b_path)

            matches_a = imread(matches_a_path)

            # Convert from RGB -> BGR
            image_a = image_a[..., [2, 1, 0]]
            image_b = image_b[..., [2, 1, 0]]
            matches_a = matches_a[..., np.newaxis]

            # Scale from [0, 255] -> [0.0, 1.0]
            image_a = image_a / 255.0
            image_b = image_b / 255.0
            matches_a = matches_a / 255.0

            # Deal with images not multiple of 64 (e.g.: Sintel, Middlebury, etc.)
            # Copied from the method from net.py: adapt_x()
            height_a, width_a, channels_a = image_a.shape
            height_b, width_b, channels_b = image_b.shape
            height_ma, width_ma, channels_ma = matches_a.shape

            # Assert matching image sizes
            assert height_a == height_b and width_a == width_b and channels_a == channels_b, (
                   "FATAL: image dimensions do not match. Image 1 has shape: {0}, Image 2 has shape: {1}".format(
                       image_a.shape, image_b.shape
                   ))
            # Assert that the mask matches dims too
            assert height_a == height_ma and width_a == width_ma, ("FATAL: mask width, height do not match the images."
                                                                   "Images have shape: {0}, mask: {1}".format(
                image_a.shape[:-1], matches_a.shape[:-1]))
            # Assert correct number of channels
            assert channels_ma == 1, ("FATAL: mask should be binary but the number of channels is not one but {0}".format(
                channels_ma
            ))

            if DEBUG:
                print("OG shapes before padding (images-matches): ")
                print("img_a: {0}\nimg_b: {1}\nmch_a: {2}".format(image_a.shape, image_b.shape, matches_a.shape))
            original_shape = image_a.shape

            if height_a % divisor != 0 or width_a % divisor != 0:
                if DEBUG:
                    print("{}x{} dimensions are not both multiple of {}, must pad...".format(height_a, width_a,
                                                                                             divisor))
                new_height = int(ceil(height_a / divisor) * divisor)
                new_width = int(ceil(width_a / divisor) * divisor)
                pad_height = new_height - height_a
                pad_width = new_width - width_a
                padding = [(0, pad_height), (0, pad_width), (0, 0)]
                if DEBUG:
                    print("Padding with: {}x{} pixels, to achieve a target size: {}x{}".format(pad_height, pad_width,
                                                                                               new_height, new_width))

                image_a = np.pad(image_a, padding, mode='constant', constant_values=0.)
                image_b = np.pad(image_b, padding, mode='constant', constant_values=0.)
                matches_a = np.pad(matches_a, padding, mode='constant', constant_values=0.)

            if DEBUG:
                print("NEW shapes after padding (images-matches): ")
                print("img_a: {0}\nimg_b: {1}\nmch_a: {2}".format(image_a.shape, image_b.shape, matches_a.shape))
                print("img_a has range: ({}, {})".format(np.min(image_a), np.max(image_a)))
                print("matches_a has range: ({}, {})".format(np.min(matches_a), np.max(matches_a)))
                print("img_b has range: ({}, {})".format(np.min(image_b), np.max(image_b)))

            image_a_raw = image_a.tostring()
            image_b_raw = image_b.tostring()
            matches_a_raw = matches_a.tostring()

            # Pad flows as well for training (to compute the loss)
            # From net.py: apply_y()
            flow = open_flo_file(flow_path)
            sparse_flow = open_flo_file(sparse_flow_path)
            flow_height, flow_width, flow_ch = flow.shape
            sparse_height, sparse_width, sparse_ch = sparse_flow.shape

            # Assert correct dimensions
            assert flow_height == sparse_height and flow_width == sparse_width and flow_ch == sparse_ch, \
                ("Fatal: both flows should have matching dimensions but instead have: out_flow.shape: {0},"
                 " sparse_flow: {1}".format(flow.shape, sparse_flow.shape))

            assert flow_ch == 2, ("Expected flow field to have 2 channels (horizontal + vertical) but it has: "
                                  "{0}".format(flow_ch))

            if DEBUG:
                print("OG shapes before padding (flows): ")
                print("sparse_flow: {0}\nflow: {1}".format(sparse_flow.shape, flow.shape))
                print("sparse_flow has range: ({}, {})".format(np.min(sparse_flow), np.max(sparse_flow)))
                print("and after normalisation: ({}, {})".format(np.min(sparse_flow)/20, np.max(sparse_flow)/20))

            # Pad if necessary (like above or to be more precise like 'apply_y' (net.py))
            if flow_height % divisor != 0 or flow_width % divisor != 0:
                new_height = int(ceil(flow_height / divisor) * divisor)
                new_width = int(ceil(flow_width / divisor) * divisor)
                pad_height = new_height - flow_height
                pad_width = new_width - flow_width
                padding = [(0, pad_height), (0, pad_width), (0, 0)]
                flow = np.pad(flow, padding, mode='constant', constant_values=0.)
                sparse_flow = np.pad(sparse_flow, padding, mode='constant', constant_values=0.)

            if DEBUG:
                print("NEW shapes after padding (flows): ")
                print("sparse_flow: {0}\nflow: {1}".format(sparse_flow.shape, flow.shape))

            # Encode as string to include in TFrecord
            flow_raw = flow.tostring()
            sparse_flow_raw = sparse_flow.tostring()

            tf_example = image_example(image_a_raw, image_b_raw, matches_a_raw, sparse_flow_raw, flow_raw)
            writer.write(tf_example.SerializeToString())
            pbar.update(count + 1)
            count += 1


def main():
    # Load train/val split into arrays
    train_val_split = np.loadtxt(FLAGS.train_val_split)
    train_idxs = np.flatnonzero(train_val_split == TRAIN)
    val_idxs = np.flatnonzero(train_val_split == VAL)

    # Convert the train and val datasets into .tfrecords format
    if 'chairs' in FLAGS.data_dir.lower() or FLAGS.dataset.lower() == 'chairs':
        print("Chosen dataset is 'FlyingChairs'")
        train_name = 'fc_train_all'
        val_name = 'fc_val_all'
        set_name = 'flying_chairs'
    elif 'things' in FLAGS.data_dir.lower() or FLAGS.dataset.lower() == 'things':
        print("Chosen dataset is 'FlyingThings3D'")
        train_name = 'ft3d_train_all'
        val_name = 'ft3d_val_all'
        set_name = 'flying_things3D'
    elif 'sintel_clean' in FLAGS.data_dir.lower() or FLAGS.dataset.lower() == 'sintel_clean':
        print("Chosen dataset is 'MPI-Sintel (clean pass)'")
        train_name = 'sintel_clean_train_all'
        val_name = 'sintel_clean_val_all'
        set_name = 'sintel_clean'
    elif 'sintel_final' in FLAGS.data_dir.lower() or FLAGS.dataset.lower() == 'sintel_final':
        print("Chosen dataset is 'MPI-Sintel (final pass)'")
        train_name = 'sintel_final_train_all'
        val_name = 'sintel_final_val_all'
        set_name = 'sintel_final'
    elif 'sintel_all' in FLAGS.data_dir.lower() or FLAGS.dataset.lower() == 'sintel_all':
        print("Chosen dataset is 'MPI-Sintel (final + clean pass)'")
        train_name = 'sintel_train_all'
        val_name = 'sintel_val_all'
        set_name = 'sintel_all'
    # Add more datasets here (to change the final tfrecords name)
    # elif 'set_name' in FLAGS.data_dir:
    else:
        print("Chosen dataset is 'FlyingChairs'")
        train_name = 'flying_chairs_train'
        val_name = 'flying_chairs_val'
        set_name = 'flying_chairs'

    # Actual conversion
    print("Generating TFRecords...")
    if FLAGS.specify_split == 'train':  # only create train split
        print("Current split: {}".format(train_name))
        convert_dataset(train_idxs, train_name, matcher=FLAGS.matcher, dataset=set_name)

    elif FLAGS.specify_split == 'val':  # only create validation split
        print("Current split: {}".format(val_name))
        convert_dataset(val_idxs, val_name, matcher=FLAGS.matcher, dataset=set_name)

    else:  # create both
        print("Current split: {}".format(train_name))
        convert_dataset(train_idxs, train_name, matcher=FLAGS.matcher, dataset=set_name)
        print("Current split: {}".format(val_name))
        convert_dataset(val_idxs, val_name, matcher=FLAGS.matcher, dataset=set_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory that includes all .png and .flo files in the dataset'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        help="Dataset name: 'chairs', 'things', 'sintel', 'kitti'",
        default='chairs'
    )
    parser.add_argument(
        '--train_val_split',
        type=str,
        required=True,
        help='Path to text file with train-validation split (1-train, 2-validation)'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Directory to output .tfrecords files'
    )
    parser.add_argument(
        '--matcher',
        type=str,
        required=False,
        help='Default matcher selected (deepmatching)',
        default='deepmatching'
    )
    parser.add_argument(
        '--specify_split',
        type=str,
        required=False,
        help='Optionally specify that we only want one split: train or val',
        default='all'
    )
    FLAGS = parser.parse_args()

    # Verify arguments are valid
    if not os.path.isdir(FLAGS.data_dir):
        raise ValueError('data_dir must exist and be a directory')
    if not os.path.isdir(FLAGS.out):
        raise ValueError('out must exist and be a directory')
    if not os.path.exists(FLAGS.train_val_split):
        raise ValueError('train_val_split must exist')
    main()
