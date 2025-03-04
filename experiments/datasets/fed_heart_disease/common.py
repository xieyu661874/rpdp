import torch
from .dataset import FedHeartDisease, HeartDiseaseRaw

NUM_CLIENTS = 4
BATCH_SIZE = 4
NUM_EPOCHS_POOLED = 50
LR = 0.001
Optimizer = torch.optim.Adam # torch.optim.SGD
FedClass = FedHeartDisease
RawClass = HeartDiseaseRaw

def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    return (486 // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
