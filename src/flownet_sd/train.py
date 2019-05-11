from ..dataloader import load_batch
from ..dataset_configs import FLYING_CHAIRS_DATASET_CONFIG, FLYING_CHAIRS_ALL_DATASET_CONFIG
from ..training_schedules import LONG_SCHEDULE
from .flownet_sd import FlowNetSD

# Create a new network
net = FlowNetSD()

# Load a batch of data
input_a, input_b, matches_a, sparse_flow, flow = load_batch(FLYING_CHAIRS_ALL_DATASET_CONFIG, 'sample', net.global_step)

# Train on the data
net.train(
    log_dir='./logs/flownet_sd_sample',
    training_schedule=LONG_SCHEDULE,
    input_a=input_a,
    input_b=input_b,
    out_flow=flow
)
