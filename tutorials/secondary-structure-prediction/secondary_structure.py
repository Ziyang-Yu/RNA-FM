import sys
import os
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import fm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import random

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(2021)