"""
Add dataset configurations here. Each dataset must have the following structure:

NAME = {
    IMAGE_HEIGHT: int,
    IMAGE_WIDTH: int,
    PADDED_IMAGE_HEIGHT: int,  # necessary since tf.slim cuts to the passed dimensions regardless of the OG input size
    PADDED_IMAGE_WIDTH: int,
    ITEMS_TO_DESCRIPTIONS: {
        'image_a': 'A 3-channel image.',
        'image_b': 'A 3-channel image.',
        'flow': 'A 2-channel optical flow field',
    },
    SIZES: {
        'train': int,
        'valid': int,    (optional)
        ...
    },
    BATCH_SIZE: int,
    PATHS: {
        'train': '',
        'valid': '', (optional)
        ...
    }
}
"""

"""
note that one step = one batch of data processed, ~not~ an entire epoch
'coeff_schedule_param': {
    'half_life': 50000,         after this many steps, the value will be i + (f - i)/2
    'initial_coeff': 0.5,       initial value
    'final_coeff': 1,           final value
},
"""

# Contains all the information necessary to deal with the two input types considered: a pair of images or first image,
# matching mask (location) + sparse flow
FLYING_CHAIRS_ALL_DATASET_CONFIG = {
    'IMAGE_HEIGHT': 384,
    'IMAGE_WIDTH': 512,
    'PADDED_IMAGE_HEIGHT': 384,
    'PADDED_IMAGE_WIDTH': 512,
    'ITEMS_TO_DESCRIPTIONS': {
        'image_a': 'A 3-channel image.',
        'image_b': 'A 3-channel image.',
        'matches_a': 'A 1-channel matching mask (1s pixels matched, 0s not matched).',
        'sparse_flow': 'A sparse flow initialised from a set of sparse matches.',
        'flow': 'A 2-channel optical flow field.',
    },
    'SIZES': {
        'train': 22232,
        'valid': 640,
    },
    'BATCH_SIZE': 8,
    'PATHS': {
        'train': '/datasets/GPI/optical_flow/TFrecords/interp/fc_train_all.tfrecord',
        'valid': '/datasets/GPI/optical_flow/TFrecords/interp/fc_val_all.tfrecord',
    },
    'PREPROCESS': {
        'scale': False,
        'crop_height': 320,
        'crop_width': 448,
        'image_a': {
            'translate': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.4,
                'prob': 1.0,
            },
            'rotate': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.4,
                'prob': 1.0,
            },
            'zoom': {
                'rand_type': "uniform_bernoulli",
                'exp': True,
                'mean': 0.2,
                'spread': 0.4,
                'prob': 1.0,
            },
            'squeeze': {
                'rand_type': "uniform_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.3,
                'prob': 1.0,
            },
            'noise': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0.03,
                'spread': 0.03,
                'prob': 1.0,
            },
        },
        # All preprocessing to image A will be applied to image B in addition to the following.
        'image_b': {
            'translate': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'rotate': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'zoom': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'gamma': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'brightness': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'contrast': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'color': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'coeff_schedule_param': {
                'half_life': 50000,
                'initial_coeff': 0.5,
                'final_coeff': 1,
            },
        }
    }
}

# FlyingThings3D
FLYING_THINGS_3D_ALL_DATASET_CONFIG = {
    'IMAGE_HEIGHT': 540,
    'IMAGE_WIDTH': 960,
    'PADDED_IMAGE_HEIGHT': 576,
    'PADDED_IMAGE_WIDTH': 960,
    'ITEMS_TO_DESCRIPTIONS': {
        'image_a': 'A 3-channel image.',
        'image_b': 'A 3-channel image.',
        'matches_a': 'A 1-channel matching mask (1s pixels matched, 0s not matched).',
        'sparse_flow': 'A sparse flow initialised from a set of sparse matches.',
        'flow': 'A 2-channel optical flow field.',
    },
    'SIZES': {
        'train': 19640,
        'valid': 3824,
    },
    'BATCH_SIZE': 4,
    'PATHS': {
        'train': '/datasets/GPI/optical_flow/TFrecords/interp/ft3d_train_all.tfrecord',
        'valid': '/datasets/GPI/optical_flow/TFrecords/interp/ft3d_val_all.tfrecord',
    },
    'PREPROCESS': {
        'scale': False,
        'crop_height': 384,
        'crop_width': 768,
        'image_a': {
            'translate': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.4,
                'prob': 1.0,
            },
            'rotate': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.4,
                'prob': 1.0,
            },
            'zoom': {
                'rand_type': "uniform_bernoulli",
                'exp': True,
                'mean': 0.2,
                'spread': 0.4,
                'prob': 1.0,
            },
            'squeeze': {
                'rand_type': "uniform_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.3,
                'prob': 1.0,
            },
            'noise': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0.03,
                'spread': 0.03,
                'prob': 1.0,
            },
        },
        # All preprocessing to image A will be applied to image B in addition to the following.
        'image_b': {
            'translate': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'rotate': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'zoom': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'gamma': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'brightness': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'contrast': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'color': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'coeff_schedule_param': {
                'half_life': 50000,
                'initial_coeff': 0.5,
                'final_coeff': 1,
            },
        }
    }
}
# Add here configs for other datasets. For instance, sintel/clean, sintel/final, slowflow, etc.
# MPI-Sintel (Final + clean pass)
SINTEL_ALL_DATASET_CONFIG = {
    'IMAGE_HEIGHT': 436,
    'IMAGE_WIDTH': 1024,
    'PADDED_IMAGE_HEIGHT': 448,
    'PADDED_IMAGE_WIDTH': 1024,
    'ITEMS_TO_DESCRIPTIONS': {
        'image_a': 'A 3-channel image.',
        'image_b': 'A 3-channel image.',
        'matches_a': 'A 1-channel matching mask (1s pixels matched, 0s not matched).',
        'sparse_flow': 'A sparse flow initialised from a set of sparse matches.',
        'flow': 'A 2-channel optical flow field.',
    },
    'SIZES': {
        'train': 9151,
        'valid': 1259,
    },
    'BATCH_SIZE': 4,
    'PATHS': {
        'train': '/datasets/GPI/optical_flow/TFrecords/interp/sintel_train_all.tfrecord',
        'valid': '/datasets/GPI/optical_flow/TFrecords/interp/sintel_val_all.tfrecord',
    },
    'PREPROCESS': {
        'scale': False,
        'crop_height': 384,
        'crop_width': 768,
        'image_a': {
            'translate': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.4,
                'prob': 1.0,
            },
            'rotate': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.4,
                'prob': 1.0,
            },
            'zoom': {
                'rand_type': "uniform_bernoulli",
                'exp': True,
                'mean': 0.2,
                'spread': 0.4,
                'prob': 1.0,
            },
            'squeeze': {
                'rand_type': "uniform_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.3,
                'prob': 1.0,
            },
            # 'noise': {
            #     'rand_type': "uniform_bernoulli",
            #     'exp': False,
            #     'mean': 0.03,
            #     'spread': 0.03,
            #     'prob': 1.0,
            # },
            # missing transformations:
            # * Luminance: 'lmult_pow', 'lmult_mult', 'lmult_add' (first multiply then add? across channels?
            # * Saturation: 'sat_pow', 'sat_mult', 'sat_add'
            # * Colour: 'col_pow', 'col_mult', 'col_add'
            # * Luminance again but changing order of operations (x/+): 'ladd_pow', 'ladd_mult', 'ladd_add'
            # * Interchange colour channels: 'col_rotate'
            # More or less compensated by the luminance, colour, gamma and contrast augmentations provided for image_b
        },
        # All preprocessing to image A will be applied to image B in addition to the following.
        'image_b': {
            'translate': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'rotate': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'zoom': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'gamma': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'brightness': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'contrast': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'color': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            # I think this is not used? It tries to define training-related learning parameters but those are defined in
            # 'training_schedules.py'
            'coeff_schedule_param': {
                'half_life': 50000,
                'initial_coeff': 0.5,
                'final_coeff': 1,
            },
        }
    }
}

SINTEL_FINAL_ALL_DATASET_CONFIG = {
    'IMAGE_HEIGHT': 436,
    'IMAGE_WIDTH': 1024,
    'PADDED_IMAGE_HEIGHT': 448,
    'PADDED_IMAGE_WIDTH': 1024,
    'ITEMS_TO_DESCRIPTIONS': {
        'image_a': 'A 3-channel image.',
        'image_b': 'A 3-channel image.',
        'matches_a': 'A 1-channel matching mask (1s pixels matched, 0s not matched).',
        'sparse_flow': 'A sparse flow initialised from a set of sparse matches.',
        'flow': 'A 2-channel optical flow field.',
    },
    'SIZES': {
        'train': 1816,
        'valid': 133,
    },
    'BATCH_SIZE': 4,
    'PATHS': {
        'train': '/datasets/GPI/optical_flow/TFrecords/interp/sintel_final_train_all.tfrecord',
        'valid': '/datasets/GPI/optical_flow/TFrecords/interp/sintel_final_val_all.tfrecord',
    },
    'PREPROCESS': {
        'scale': False,
        'crop_height': 384,
        'crop_width': 768,
        'image_a': {
            'translate': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.4,
                'prob': 1.0,
            },
            'rotate': {
                'rand_type': "uniform_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.4,
                'prob': 1.0,
            },
            'zoom': {
                'rand_type': "uniform_bernoulli",
                'exp': True,
                'mean': 0.2,
                'spread': 0.4,
                'prob': 1.0,
            },
            'squeeze': {
                'rand_type': "uniform_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.3,
                'prob': 1.0,
            },
            # 'noise': {
            #     'rand_type': "uniform_bernoulli",
            #     'exp': False,
            #     'mean': 0.03,
            #     'spread': 0.03,
            #     'prob': 1.0,
            # },
            # missing transformations:
            # * Luminance: 'lmult_pow', 'lmult_mult', 'lmult_add' (first multiply then add? across channels?
            # * Saturation: 'sat_pow', 'sat_mult', 'sat_add'
            # * Colour: 'col_pow', 'col_mult', 'col_add'
            # * Luminance again but changing order of operations (x/+): 'ladd_pow', 'ladd_mult', 'ladd_add'
            # * Interchange colour channels: 'col_rotate'
            # More or less compensated by the luminance, colour, gamma and contrast augmentations provided for image_b
        },
        # All preprocessing to image A will be applied to image B in addition to the following.
        'image_b': {
            'translate': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'rotate': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'zoom': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.03,
                'prob': 1.0,
            },
            'gamma': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'brightness': {
                'rand_type': "gaussian_bernoulli",
                'exp': False,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'contrast': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            'color': {
                'rand_type': "gaussian_bernoulli",
                'exp': True,
                'mean': 0,
                'spread': 0.02,
                'prob': 1.0,
            },
            # I think this is not used? It tries to define training-related learning parameters but those are defined in
            # 'training_schedules.py'
            'coeff_schedule_param': {
                'half_life': 50000,
                'initial_coeff': 0.5,
                'final_coeff': 1,
            },
        }
    }
}
