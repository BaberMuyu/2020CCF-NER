import os
import torch
import numpy as np
import random

ROOT_PROJECT = os.path.abspath(__file__)[:-len("global_config.py")]

ROOT_LOG = ROOT_PROJECT + "log/"
ROOT_DATA = ROOT_PROJECT + "data/"
ROOT_UTIL = ROOT_PROJECT + "util/"
ROOT_MODEL = ROOT_PROJECT + "model/"
ROOT_MODULE = ROOT_PROJECT + "module/"
ROOT_RESULT = ROOT_PROJECT + "result/"
ROOT_WEIGHT = ROOT_PROJECT + "weight/"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed = 20
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

root_list = [ROOT_LOG, ROOT_DATA, ROOT_UTIL, ROOT_MODEL, ROOT_MODULE, ROOT_RESULT, ROOT_WEIGHT]
for root_dir in root_list:
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
