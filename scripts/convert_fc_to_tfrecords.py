import argparse
import os
import sys
import numpy as np
from progressbar import ProgressBar, Percentage, Bar
from scipy.misc import imread
import tensorflow as tf
from math import ceil

FLAGS = None

# Values defined here: https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs
TRAIN = 1
VAL = 2


# https://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
def open_flo_file(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
            # Reshape data into 3D array (columns, rows, bands)
            return np.resize(data, (w[0], h[0], 2))


# https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/convert_to_records.py
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# TODO: add support for images with sizes not multiple of 64 (padding)
# TODO: create a subfolder in sintel/clean and sintel/final 'flatten' that contains an alphabetical list of files
# to ease the creation of the tfrecord and the general use of the framework
# Cropping during training is not necessary since the flow with also be padded with 0s and no errors will be added!
def convert_dataset(indices, name, input_type='image_pairs', dataset='flying_chairs', divisor=64):
    # Open a TFRRecordWriter
    filename = os.path.join(FLAGS.out, name + '.tfrecords')
    writeOpts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(filename, options=writeOpts)

    # Load each data sample (image_a, image_b, flow) and write it to the TFRecord
    count = 0
    pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(indices)).start()
    for i in indices:
        if input_type == 'image_pairs':  # input to the network is: frame0 + frame1
            if dataset == 'flying_chairs':
                image_a_path = os.path.join(FLAGS.data_dir, '{0:05d}_img1.ppm'.format(i + 1))  #'%05d_img1.ppm' % (i + 1))
                image_b_path = os.path.join(FLAGS.data_dir, '{0:05d}_img2.ppm'.format(i + 1))  #'%05d_img2.ppm' % (i + 1))
                flow_path = os.path.join(FLAGS.data_dir, '{0:05d}_flow.flo'.format(i + 1))  # '%05d_flow.flo' % (i + 1))
            elif dataset == 'sintel_clean':
                data_subdir = 'clean/flatten'
                image_a_path = os.path.join(FLAGS.data_dir, data_subdir, 'frame_{0:04d}.png'.format(i))
                image_b_path = os.path.join(FLAGS.data_dir, data_subdir, 'frame_{0:04d}.png'.format(i + 1))
                flow_path = os.path.join(FLAGS.data_dir, data_subdir, '{0:04d}_flow.flo'.format(i))
            elif dataset == 'sintel_final':
                data_subdir = 'final/flatten'
                image_a_path = os.path.join(FLAGS.data_dir, data_subdir, 'frame_{0:04d}.png'.format(i))
                image_b_path = os.path.join(FLAGS.data_dir, data_subdir, 'frame_{0:04d}.png'.format(i + 1))
                flow_path = os.path.join(FLAGS.data_dir, data_subdir, '{0:04d}_flow.flo'.format(i))
            # Add more datasets here
            # elif dataset == 'another_of_dataset':
            else:
                raise ValueError("Invalid dataset name!")

        elif input_type == 'image_matches':  # input to the network is: frame0 + matches(frame0, frame1)
            if dataset == 'flying_chairs':
                image_a_path = os.path.join(FLAGS.data_dir, '{0:05d}_img1.ppm'.format(i + 1))
                image_b_path = os.path.join(FLAGS.data_dir, '{0:05d}_img1_mask.ppm'.format(i + 1))
                flow_path = os.path.join(FLAGS.data_dir, '{0:05d}_flow.flo'.format(i + 1))
            elif dataset == 'sintel_clean':
                data_subdir = 'clean/flatten'
                image_a_path = os.path.join(FLAGS.data_dir, data_subdir, 'frame_{0:04d}.png'.format(i))
                image_b_path = os.path.join(FLAGS.data_dir, data_subdir, 'frame_{0:04d}_mask.png'.format(i))
                flow_path = os.path.join(FLAGS.data_dir, data_subdir, '{0:04d}_flow.flo'.format(i))
            elif dataset == 'sintel_final':
                data_subdir = 'final/flatten'
                image_a_path = os.path.join(FLAGS.data_dir, data_subdir, 'frame_{0:04d}.png'.format(i))
                image_b_path = os.path.join(FLAGS.data_dir, data_subdir, 'frame_{0:04d}_mask.png'.format(i))
                flow_path = os.path.join(FLAGS.data_dir, data_subdir, '{0:04d}_flow.flo'.format(i))
            # Add more datasets here
            # elif dataset == 'another_of_dataset':
            else:
                raise ValueError("Invalid dataset name!")

        image_a = imread(image_a_path)
        image_b = imread(image_b_path)

        # Convert from RGB -> BGR
        image_a = image_a[..., [2, 1, 0]]
        if input_type == 'image_pairs':
            image_b = image_b[..., [2, 1, 0]]
        else:
            image_b = image_b[..., np.newaxis]

        # Scale from [0, 255] -> [0.0, 1.0]
        image_a = image_a / 255.0
        image_b = image_b / 255.0

        # Deal with images not multiple of 64 (e.g.: Sintel, Middlebury, etc.)
        # Copied from the method from net.py: apply_x()
        height_a, width_a, channels_a = image_a.shape  # temporal hack so it works with any size (batch or not)
        height_b, width_b, channels_b = image_b.shape
        if not image_a.shape[-1] == image_b.shape[-1]:
            assert (channels_b == 1)  # valid (reshaped) B&W mask
        else:
            assert (channels_a == channels_b)  # images in the same colourspace!
        # Assert matching width and height
        assert (height_a == height_b and width_a == width_b)

        if height_a % divisor != 0 or width_a % divisor != 0:
            new_height = int(ceil(height_a / divisor) * divisor)
            new_width = int(ceil(width_a / divisor) * divisor)
            pad_height = new_height - height_a
            pad_width = new_width - width_a
            padding = [(0, pad_height), (0, pad_width), (0, 0)]

            image_a = np.pad(image_a, padding, mode='constant', constant_values=0.)
            image_b = np.pad(image_b, padding, mode='constant', constant_values=0.)

        image_a_raw = image_a.tostring()
        image_b_raw = image_b.tostring()

        # Must pad the flow too before saving it to string (??)
        # From net.py: apply_y()
        flow_raw = open_flo_file(flow_path)
        # Assert it is a valid flow
        assert (flow_raw.shape[-1] == 2)
        if height_a % divisor != 0 or width_a % divisor != 0:
            new_height = int(ceil(height_a / divisor) * divisor)
            new_width = int(ceil(width_a / divisor) * divisor)
            pad_height = new_height - height_a
            pad_width = new_width - width_a
            padding = [(0, pad_height), (0, pad_width), (0, 0)]
            flow_raw = np.pad(flow_raw, padding, mode='constant', constant_values=0.)

        # Encode flow as string
        flow_raw = flow_raw.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_a': _bytes_feature(image_a_raw),
            'image_b': _bytes_feature(image_b_raw),
            'flow': _bytes_feature(flow_raw)}))
        writer.write(example.SerializeToString())
        pbar.update(count + 1)
        count += 1
    writer.close()


def main():
    # Load train/val split into arrays
    train_val_split = np.loadtxt(FLAGS.train_val_split)
    train_idxs = np.flatnonzero(train_val_split == TRAIN)
    val_idxs = np.flatnonzero(train_val_split == VAL)

    # Convert the train and val datasets into .tfrecords format
    if 'flying' in FLAGS.data_dir:
        train_name = 'fc_train'
        val_name = 'fc_val'
        set_name = 'flying_chairs'
    elif 'sintel_clean' in FLAGS.data_dir:
        train_name = 'sintel_clean_train'
        val_name = 'sintel_clean_val'
        set_name = 'sintel_clean'
    elif 'sintel_final' in FLAGS.data_dir:
        train_name = 'sintel_final_train'
        val_name = 'sintel_final_val'
        set_name = 'sintel_final'
    # Add more datasets here (to change the final tfrecords name)
    # elif 'set_name' in FLAGS.data_dir:
    else:
        train_name = 'flying_chairs_train'
        val_name = 'flying_chairs_val'
        set_name = 'flying_chairs'

    # Actual conversion
    convert_dataset(train_idxs, train_name, input_type=FLAGS.input_type, dataset=set_name)
    convert_dataset(val_idxs, val_name, input_type=FLAGS.input_type, dataset=set_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory that includes all .ppm and .flo files in the dataset'
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
        help='Directory for output .tfrecords files'
    )
    parser.add_argument(
        '--input_type',
        type=str,
        required=False,
        help='Whether to use two consecutive frames as input or one frame and the matches "towards" the next one'
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
