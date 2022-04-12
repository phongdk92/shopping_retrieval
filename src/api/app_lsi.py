#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/02/2022

@author: phongdk
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/02/2022

@author: phongdk
"""

import glob
import os
import logging.config
import pickle
import random
import time
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from top2vec import Top2Vec
from icecream import ic
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel


from gensim.models.phrases import Phrases
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity
from gensim import corpora
from src.utils import utils
from src.utils.utils import detect_branch, detect_product_code

logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)
DATA_DIR = os.environ.get("DATA_DIR")
random.seed(123)
np.random.seed(123)

app = FastAPI()
PROMETHEUS_PORT = int(os.environ.get('PROMETHEUS_PORT', 8010))
# DISTANCE_THRESHOLD = 0.08
SIMILARITY_THRESHOLD = 0.9
TOP_K = 20
# doc2vec_model = Top2Vec.load(config.DOC2VEC_FILENAME).model
# indexer = pickle.load(open(config.INDEXER_FILENAME, 'rb'))

# data_sample = pd.read_pickle(f"{DATA_DIR}/data/shopee_sample.pkl").set_index('Product ID')
data_sample = pd.read_pickle(f"{DATA_DIR}/data/shopee_sample.pkl")
model_name = f"{DATA_DIR}/models/lsi.model"
phrase_model_name = f"{DATA_DIR}/models/phrase.model"
dictionary_filename = f"{DATA_DIR}/models/dictionary.pkl"
matrix_similarity_indexer = f"{DATA_DIR}/models/matrix_similarity.model"

lsi_model = LsiModel.load(model_name)
phrase_model = Phrases.load(phrase_model_name)
dictionary = corpora.Dictionary().load(dictionary_filename)
index = MatrixSimilarity.load(matrix_similarity_indexer)


class Content(BaseModel):
    title: Optional[str] = None


@app.get('/api/health')
def health_check():
    return JSONResponse(content=jsonable_encoder({"status": "ok"}))


@app.post('/v1.0/documents/search-by-text')
async def documents_search_by_text(content: Content):
    logger.info("**********************")
    title = utils.clean_text(content.title.lower()).split()
    ic(title)
    title = phrase_model[title]
    ic(title)
    vec_bow = dictionary.doc2bow(title)
    vec_lsi = lsi_model[vec_bow]  # convert the query to LSI space
    ic(vec_lsi[:10])
    sims = index[vec_lsi]  # perform a similarity query against the corpus
    # ic(list(enumerate(sims[:10])))  # print (document_number, document_similarity) 2-tuples
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    candidates = []
    for doc_position, doc_score in sims[:TOP_K]:
        if doc_score > SIMILARITY_THRESHOLD:
            candidates.append(doc_position)
    candidates = np.array(candidates).tolist()
    selection = candidates
    # branch = detect_branch(text=title)
    # product_code = detect_product_code(text=title)
    #
    # candidates = []
    # for l, d in zip(labels[0], distances[0]):
    #     if d < DISTANCE_THRESHOLD:
    #         candidates.append(l)
    #     else:
    #         break
    # if branch:
    #     pass
    # candidates = np.array(candidates).tolist()
    # ic(candidates)
    # selection = []
    # if product_code:
    #     ic(product_code)
    #     df_candidates = data_sample.loc[candidates]
    #     ic(df_candidates.shape)
    #     for row in df_candidates.itertuples():
    #         c_product_code = detect_product_code(text=row.title)
    #         ic(row.title, c_product_code)
    #         if set(product_code).intersection(set(c_product_code)):
    #             selection.append(row.Index)
    # else:
    #     selection = candidates
    # # return JSONResponse(content=jsonable_encoder({"candidates": candidates}))
    # ic("test")
    df_candidates = data_sample.loc[selection]
    ic("test1")
    # return selection
    result = df_candidates.to_dict(orient='index')#['title']
    ic(result)
    return result

@app.middleware("http")
async def log_request(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # milliseconds
    # REQUEST_COUNTER.labels(method=request.method,
    #                        endpoint=request.url.path.split("=")[0],
    #                        http_status=response.status_code).inc()
    # REQUEST_LATENCY.labels(method=request.method,
    #                        endpoint=request.url.path.split("=")[0],
    #                        http_status=response.status_code).observe(process_time)
    return response
