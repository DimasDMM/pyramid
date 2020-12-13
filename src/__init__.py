import numpy as np
np.random.seed(42)

def setup_logger(logger, to_file=False):
    if to_file:
        logger.basicConfig(filename='./logger.log',
                           filemode='a',
                           format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                           datefmt='%H:%M:%S',
                           level=logger.DEBUG)
    else:
        logger.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                           datefmt='%H:%M:%S',
                           level=logger.DEBUG)
