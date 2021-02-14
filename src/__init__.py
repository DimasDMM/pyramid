import numpy as np
import os
import random
import torch

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DEFAULT_SEED = 42

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_project_path(*paths):
    return os.path.join(BASE_PATH, *paths)

def setup_logger(logger, output_file=None):
    if output_file is not None:
        logger.basicConfig(filename=get_project_path(output_file),
                           filemode='a',
                           format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                           datefmt='%H:%M:%S',
                           level=logger.DEBUG)
    else:
        logger.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                           datefmt='%H:%M:%S',
                           level=logger.DEBUG)

def none_or_str(value):
    if value == 'None':
        return None
    return value

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

# Default set up
set_seed(DEFAULT_SEED)
