#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/02/2022

@author: phongdk
"""

import logging.config
import pickle

import hnswlib

from src import config
from src.utils.utils import save_to_pickle

logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

MAX_NUMBER_ELEMENTS = 20_000_000


class FastApproximateNearestNeighbors:
    def __init__(self, num_threads=-1):
        self.ef_construction = 200 * 2
        self.M = 16 * 2
        self.space = 'cosine'
        self.num_threads = num_threads

    def index_data(self, embeddings, ids):
        logger.info(f"Data shape: {embeddings.shape}")
        dim = embeddings.shape[1]
        # Generating sample data
        # Declaring index
        logger.info(f"Index data with space: {self.space}, dim: {dim}")
        self.p = hnswlib.Index(space=self.space,
                               dim=dim)  # d = 1.0 - sum(Ai*Bi) / sqrt(sum(Ai*Ai) * sum(Bi*Bi))

        # Initializing index - the maximum number of elements should be known beforehand
        self.p.init_index(max_elements=MAX_NUMBER_ELEMENTS,
                          ef_construction=self.ef_construction,
                          M=self.M)
        logger.info("Add items to indexer")
        # Element insertion (can be called several times):
        # self.p.add_items(embeddings, ids)
        self.add_data(embeddings=embeddings,
                      ids=ids)
        # Controlling the recall by setting ef:
        self.p.set_ef(50)  # ef should always be > k

    def add_data(self, embeddings, ids):
        self.p.add_items(embeddings, ids, num_threads=self.num_threads)

    def save(self, filename):
        save_to_pickle(data=self.p, filename=filename)

    def load(self, filename):
        return pickle.load(open(filename, 'rb'))

    def knn_query(self, data, k=10):
        # Query dataset, k - number of closest elements (returns 2 numpy arrays)
        labels, distances = self.p.knn_query(data=data,
                                             k=k,
                                             num_threads=-1)
        return labels, distances
