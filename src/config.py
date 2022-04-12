#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/03/2021

@author: phongdk
"""
import os
from datetime import datetime
import icecream

DATA_DIR = os.getenv('DATA_DIR', '/shared_storage/bi_mlearn_training/coccoc_shopping')
DATA_FILENAME = f'{DATA_DIR}/data/shopee_sample.pkl'
DOC2VEC_FILENAME = f"{DATA_DIR}/models/top2vec_2M_learn.model"
INDEXER_FILENAME = f"{DATA_DIR}/models/indexer.pkl"


def time_format():
    return f'{datetime.now()}   |>  '


"""
CONFIG DEBUG MODE -> other files just import config
"""
icecream.ic.configureOutput(prefix=time_format,
                            includeContext=True)
icecream.install()

"""
CONFIG LOG FORMAT
"""
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(filename)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        'gensim': {
            'handlers': ['default'],
            'level': 'ERROR',
            'propagate': False
        },
        'apscheduler': {
            'handlers': ['default'],
            'level': 'ERROR',
            'propagate': False
        },
        '__main__': {  # if __name__ == '__main__'
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
    }
}
