#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/02/2022

@author: phongdk
"""

import os
import logging.config
import pickle
import random
from typing import Optional

import numpy as np
import pandas as pd
from gensim.models import Phrases
from top2vec import Top2Vec
from src import config
from icecream import ic
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.utils import utils
from src.utils.utils import detect_branch, detect_product_code

logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)
DATA_DIR = os.environ.get("DATA_DIR")
random.seed(123)
np.random.seed(123)

app = FastAPI()
PROMETHEUS_PORT = int(os.environ.get('PROMETHEUS_PORT', 8010))
DISTANCE_THRESHOLD = 0.45
TOP_K = 10
doc2vec_model = Top2Vec.load(config.DOC2VEC_FILENAME).model
indexer = pickle.load(open(config.INDEXER_FILENAME, 'rb'))
# phrase_model_name = f"{DATA_DIR}/models/phrase.model"
# phrase_model = Phrases.load(phrase_model_name)

data_sample = pd.read_pickle(f"{DATA_DIR}/data/shopee_sample.pkl").set_index('Product ID')


class Content(BaseModel):
    title: Optional[str] = None


@app.get('/api/health')
def health_check():
    return JSONResponse(content=jsonable_encoder({"status": "ok"}))


@app.post('/v1.0/documents/search-by-text')
async def documents_search_by_text(content: Content):
    logger.info("**********************")
    title = utils.clean_text(content.title.lower())
    ic(title)
    # title = phrase_model[title]
    # ic(phrase_model[utils.process_text(title)])
    # vector = doc2vec_model.infer_vector(phrase_model[utils.process_text(title)],
    #                                     alpha=0.025,
    #                                     min_alpha=0.01,
    #                                     epochs=100)
    # ic(phrase_model[utils.process_text(title)])
    vector = doc2vec_model.infer_vector(utils.process_text(title),
                                        alpha=0.025,
                                        min_alpha=0.01,
                                        epochs=100)
    title = utils.tokenize(title)
    ic(title)
    # branch = detect_branch(text=title)
    product_code = None # detect_product_code(text=title)
    # ic(branch, product_code)
    # Query dataset, k - number of closest elements (returns 2 numpy arrays)
    labels, distances = indexer.knn_query(vector, k=TOP_K)
    ic(distances)
    # candidates = np.array([l for l, d in zip(labels[0], distances[0]) if d < DISTANCE_THRESHOLD])
    candidates = []
    for l, d in zip(labels[0], distances[0]):
        if d < DISTANCE_THRESHOLD:
            candidates.append(l)
        else:
            break
    candidates = np.array(candidates).tolist()
    ic(candidates)
    selection = []
    if product_code:
        ic(product_code)
        df_candidates = data_sample.loc[candidates]
        ic(df_candidates.shape)
        for row in df_candidates.itertuples():
            c_product_code = detect_product_code(text=row.title)
            ic(row.title, c_product_code)
            if set(product_code).intersection(set(c_product_code)):
                selection.append(row.Index)
    else:
        selection = candidates
    df_candidates = data_sample.loc[selection]
    result = df_candidates.to_dict(orient='index')
    ic(result)
    return result
