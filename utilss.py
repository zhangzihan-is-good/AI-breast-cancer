import yaml
import torch
import numpy as np
import random

def randomseed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)