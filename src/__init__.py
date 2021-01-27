import numpy as np
np.random.seed(42)

def setup_logger(logger, output_file=None):
    if output_file is not None:
        logger.basicConfig(filename=output_file,
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
