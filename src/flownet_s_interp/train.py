from ..dataloader import load_batch
from ..dataset_configs import FLYING_CHAIRS_DATASET_CONFIG
from ..training_schedules import LONG_SCHEDULE
from .flownet_s_interp import FlowNetS_interp

# Create a new network
net = FlowNetS_interp()

# Load a batch of data
input_a, matches_a, flow = load_batch(FLYING_CHAIRS_DATASET_INTERP_CONFIG, 'sample', net.global_step)

# Train on the data
net.train(
    log_dir='./logs/flownet_s_interp_sample',
    training_schedule=LONG_SCHEDULE,
    input_a=input_a,
    input_b=matches_a,
    flow=flow
)
