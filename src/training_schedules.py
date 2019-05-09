# TODO: also add proper fine-tuning for each dataset (following PWC-Net+ findings!)
# thanks to: https://arxiv.org/pdf/1809.05571.pdf (the code has not been updated yet but the paper is out!)
# schedules reproduced (aside from long which was already provided) from og repo at: github.com/lmb-freiburg/flownet2
# Must navigate to models and download models to get the text files (*proto.txt): one for 'long', 'fine' and 'short'
LONG_SCHEDULE = {
    'step_values': [400000, 600000, 800000, 1000000],
    'learning_rates': [0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 1200000,
}
# og repo adds a stepvalue at 500k iters but it is useless since it is equal to maxiter!
FINETUNE_SCHEDULE = {
    'step_values': [200000, 300000, 400000],  # might not work if the optimizer reloads previous global step!
    'learning_rates': [0.00001, 0.000005, 0.0000025, 0.00000125],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 500000,  # may need to change it to 1.7M iters (continuing Slong)
}

# Added original FlowNet 1.0 schedule (in case we need to compare results). It performed considerably worse (but faster)
SHORT_SCHEDULE = {
    'step_values': [300000, 400000, 500000],
    'learning_rates': [0.0001, 0.00005, 0.000025, 0.0000125],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 600000,
}

# TODO: change default values (copied from Sshort)
# Add learning rate disruptions to fine-tune on Sintel and Kitti from PWC-Net+ (from paper:
#  Models matter, so does training: an empirical study of CNNs for optical flow estimation"
FINETUNE_SINTEL = {
    'step_values': [300000, 400000, 500000],
    'learning_rates': [0.0001, 0.00005, 0.000025, 0.0000125],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 600000,
}

FINETUNE_KITTI = {
    'step_values': [300000, 400000, 500000],
    'learning_rates': [0.0001, 0.00005, 0.000025, 0.0000125],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 600000,
}

# Uses mixture of Sintel, KITTI and HD1K to fine-tune
FINETUNE_ROB = {
    'step_values': [300000, 400000, 500000],
    'learning_rates': [0.0001, 0.00005, 0.000025, 0.0000125],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 600000,
}