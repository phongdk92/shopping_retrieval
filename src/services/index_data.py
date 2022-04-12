#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/02/2022

@author: phongdk
"""
import os
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd
from gensim.models import Phrases
from icecream import ic
from pandarallel import pandarallel
from top2vec import Top2Vec

from src import config
from src.services.indexer import FastApproximateNearestNeighbors
from src.utils import utils

N_JOBS = 24
pandarallel.initialize(nb_workers=N_JOBS)
DATA_DIR = os.environ.get("DATA_DIR")
random.seed(123)

speed = 'learn'
# filename = f'{DATA_DIR}/data/shopee_sample.pkl'
doc2vec_model = Top2Vec.load(config.DOC2VEC_FILENAME).model
# phrase_model_name = f"{DATA_DIR}/models/phrase.model"
# phrase_model = Phrases.load(phrase_model_name)


def load_data(filename):
    ic(filename)
    # df = pd.read_csv(filename, usecols=['Product ID', 'title'], nrows=1_000_00)
    df = pd.read_pickle(filename)
    df['title'] = df['title'].parallel_apply(lambda x: utils.process_text(utils.clean_text(x)))
    ic(df.head())
    # lol_tokens = phrase_model[df['title'].values]
    return [" ".join(tokens) for tokens in lol_tokens], df['Product ID'].values
    # return lol_tokens, df['Product ID'].values


def simulate_tokenizer(x):
    return x.split()


def get_embedding_vector(tokens):
    return doc2vec_model.infer_vector(tokens,
                                      alpha=0.025,
                                      min_alpha=0.01,
                                      epochs=100)


if __name__ == '__main__':
    lol_tokens, ids = load_data(config.DATA_FILENAME)
    ic("Convert text to vectors")
    with Pool(processes=N_JOBS) as pool:
        embeddings = pool.map(get_embedding_vector, lol_tokens)
    embeddings = np.float16(np.array(embeddings))
    ic(embeddings[0].shape)
    fann = FastApproximateNearestNeighbors(num_threads=N_JOBS)
    fann.index_data(embeddings=embeddings, ids=ids)
    fann.save(filename=config.INDEXER_FILENAME)
