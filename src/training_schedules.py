# import numpy as np
# thanks to: https://arxiv.org/pdf/1809.05571.pdf (the code has not been updated yet but the paper is out!)
# schedules reproduced (aside from long which was already provided) from og repo at: github.com/lmb-freiburg/flownet2
# Must navigate to models and download models to get the text files (*proto.txt): one for 'long', 'fine' and 'short'
# step_values = list(np.round(np.linspace(1, 2999, 50)).astype('int'))
# learning_rates = []
# lr = 1e-10
# for step in step_values:
#     lr = lr * 1.58
#     learning_rates.append(lr)
# learning_rates.insert(0, 1e-10)

LR_RANGE_TEST = {  # steps and learning rates defined by a step-wise curve with (exponential) decay (increasing)
    # 'step_values': step_values,
    # 'learning_rates': learning_rates,
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0,  # remove it as we are ONLY exponentially trying diff. lrs
    'max_iter': 9838,  # around 30 minutes (aprox. 5 iters/s), enough to diverge probably (intended!)
}
LONG_SCHEDULE = {
    'step_values': [400000, 600000, 800000, 1000000],
    'learning_rates': [0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 1200000,
}
# og repo adds a stepvalue at 500k iters but it is useless since it is equal to maxiter!
# Resetting global_step after Slong so this step numbers are correct (started from 0)
FINE_SCHEDULE = {
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

# learning rate disruptions to fine-tune on Sintel and Kitti from PWC-Net+ (from paper:
#  Models matter, so does training: an empirical study of CNNs for optical flow estimation"
# The authors recently uploaded the caffe training protocols on:
# github.com/NVlabs/PWC-Net/tree/master/Caffe/model/PWC-Net_plus (although only for KITTI and Sintel it seems...)
# NOTE: they train in several stages (5 per model). Careful with overfitting as we use a smaller model (FlowNetS).
# According to the paper, they only apply 5 learning rate disruptions while training on the mixed batch with samples
# from kitti, sintel, hd1k (ROB challenge). For Sintel alone, only 2 disruptions with different lr occur
# (opened an issue to clarify this)
FINETUNE_SINTEL_S1 = {
    'step_values': [45000, 65000, 85000, 95000, 97500, 100000, 110000, 120000, 130000, 140000],
    'learning_rates': [5e-05, 2.5e-05, 1.25e-05, 6.25e-06, 3.125e-06, 1.5625e-06, 7.8125e-07, 3.90625e-07, 1.953125e-07,
                       9.765625e-08, 4.8828125e-08],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 150000,
}

FINETUNE_SINTEL_S2 = {
    'step_values': [195000, 215000, 235000, 245000, 247500, 250000, 260000, 270000, 280000, 290000],
    'learning_rates': [3e-05, 1.5e-05, 7.5e-06, 3.75e-06, 1.875e-06, 9.375e-07, 4.6875e-07, 2.34375e-07, 1.171875e-07,
                       5.859375e-08, 2.9296875e-08],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 300000,
}
# TODO: the fix worked so the global step is correctly resumed (change the other fine-tuning schedules accordingly)
# NOTE: this means that if we want to do some of this fine-tuning in a different order, we have to:
# * enable to restart from 0 or create a new config(!)
# * max_iter MUST also be changed as it is evaluated as an absolute value based of global_step (not relative to
# the new training)
# * IMPORTANT: reset checkpoint name - global_step to 0 BEFORE fine-tuning for this to work (after Sfine too!)
# E.g.: 'path/to/model/model-543303.* to path/to/model/model-0.* (change all files: meta, data and ckpt)

FINETUNE_SINTEL_S3 = {
    'step_values': [345000, 365000, 385000, 395000, 397500, 400000, 410000, 420000, 430000, 440000],
    'learning_rates': [2e-05, 1e-05, 5e-06, 2.5e-06, 1.25e-06, 6.25e-07, 3.125e-07, 1.5625e-07, 7.8125e-08, 3.90625e-08,
                       1.953125e-08],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 450000,
}

FINETUNE_SINTEL_S4 = {
    'step_values': [495000, 515000, 535000, 545000, 547500, 550000, 560000, 570000, 580000, 590000],
    'learning_rates': [1e-05, 5e-06, 2.5e-06, 1.25e-06, 6.25e-07, 3.125e-07, 1.5625e-07, 7.8125e-08, 3.90625e-08,
                       1.953125e-08, 9.765625e-09],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 600000,
}

FINETUNE_SINTEL_S5 = {
    'step_values': [645000, 665000, 685000, 695000, 697500, 700000, 710000, 720000, 730000, 740000],
    'learning_rates': [5e-06, 2.5e-06, 1.25e-06, 6.25e-07, 3.125e-07, 1.5625e-07, 7.8125e-08, 3.90625e-08, 1.953125e-08,
                       9.765625e-09, 4.8828125e-09],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 750000,
}

# concatenate all stages into only one (may fix reloading from checkpoint "problems")
# CORRECTION: this is not valid since the weight decay will NOT have different value than expected
# since it is applied only at the start and gradually changes instead of restarting it after each stage!
# That is why in the original training schedules they have separate running scripts and proto.txt files
# TODO: remove this step for the aforementioned reasons
FINETUNE_SINTEL_ALL_STAGES = {
    'step_values': [45000, 65000, 85000, 95000, 97500, 100000, 110000, 120000, 130000, 140000, 150000, 195000, 215000,
                    235000, 245000, 247500, 250000, 260000, 270000, 280000, 290000, 300000, 345000, 365000, 385000,
                    395000, 397500, 400000, 410000, 420000, 430000, 440000, 450000, 495000, 515000, 535000, 545000,
                    547500, 550000, 560000, 570000, 580000, 590000, 600000, 645000, 665000, 685000, 695000, 697500,
                    700000, 710000, 720000, 730000, 740000],
    'learning_rates': [5e-05, 2.5e-05, 1.25e-05, 6.25e-06, 3.125e-06, 1.5625e-06, 7.8125e-07, 3.90625e-07, 1.953125e-07,
                       9.765625e-08, 4.8828125e-08, 3e-05, 1.5e-05, 7.5e-06, 3.75e-06, 1.875e-06, 9.375e-07, 4.6875e-07,
                       2.34375e-07, 1.171875e-07, 5.859375e-08, 2.9296875e-08, 2e-05, 1e-05, 5e-06, 2.5e-06, 1.25e-06,
                       6.25e-07, 3.125e-07, 1.5625e-07, 7.8125e-08, 3.90625e-08, 1.953125e-08, 1e-05, 5e-06, 2.5e-06,
                       1.25e-06, 6.25e-07, 3.125e-07, 1.5625e-07, 7.8125e-08, 3.90625e-08, 1.953125e-08, 9.765625e-09,
                       5e-06, 2.5e-06, 1.25e-06, 6.25e-07, 3.125e-07, 1.5625e-07, 7.8125e-08, 3.90625e-08, 1.953125e-08,
                       9.765625e-09, 4.8828125e-09],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 750000,
}

# KITTI
FINETUNE_KITTI_S1 = {
    'step_values': [45000, 65000, 85000, 95000, 97500, 100000, 110000, 120000, 130000, 140000],
    'learning_rates': [4e-05, 2e-05, 1e-05, 5e-06, 2.5e-06, 1.25e-06, 6.25e-07, 3.125e-07, 1.5625e-07, 7.8125e-08,
                       3.90625e-08],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 150000,
}

FINETUNE_KITTI_S2 = {
    'step_values': [195000, 215000, 235000, 245000, 247500, 250000, 260000, 270000, 280000, 290000],
    'learning_rates': [4e-05, 2e-05, 1e-05, 5e-06, 2.5e-06, 1.25e-06, 6.25e-07, 3.125e-07, 1.5625e-07, 7.8125e-08,
                       3.90625e-08],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 300000,
}

FINETUNE_KITTI_S3 = {
    'step_values': [345000, 365000, 385000, 395000, 397500, 400000, 410000, 420000, 430000, 440000],
    'learning_rates': [2e-05, 1e-05, 5e-06, 2.5e-06, 1.25e-06, 6.25e-07, 3.125e-07, 1.5625e-07, 7.8125e-08, 3.90625e-08,
                       1.953125e-08],
    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,
    'max_iter': 450000,
}

FINETUNE_KITTI_S4 = {
    'step_values': [495000, 515000, 535000, 545000, 547500, 550000, 560000, 570000, 580000, 590000],
    'learning_rates': [1e-05, 5e-06, 2.5e-06, 1.25e-06, 6.25e-07, 3.125e-07, 1.5625e-07, 7.8125e-08, 3.90625e-08,
                       1.953125e-08, 9.765625e-09],
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