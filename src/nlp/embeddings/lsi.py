#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/02/2022

@author: phongdk
"""

import pickle
import random
from top2vec import Top2Vec
import pandas as pd
import os
import logging.config
from icecream import ic
from gensim.models.phrases import Phrases
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity
from gensim import corpora
from src.utils import utils
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=16)
# logging.config.dictConfig(config.LOGGING_CONFIG)
# logger = logging.getLogger(__name__)
DATA_DIR = os.environ.get("DATA_DIR")
random.seed(123)


def load_stopwords():
    df = pd.read_csv(stopwords_filename, header=None)
    return set(df[0].values)


def load_data(filename):
    ic(filename)
    df = pd.read_pickle(filename).head(1_000_000)
    ic(df.head())
    df['title'] = df['title'].parallel_apply(lambda x: utils.process_text(utils.clean_text(x)))
    # df['title'] = df['title'].parallel_apply(lambda x: utils.clean_text(x).lower().split())
    return df['title'].values


if __name__ == '__main__':
    filename = f'{DATA_DIR}/data/shopee_sample.pkl'
    model_name = f"{DATA_DIR}/models/lsi.model"
    phrase_model_name = f"{DATA_DIR}/models/phrase.model"
    dictionary_filename = f"{DATA_DIR}/models/dictionary.pkl"
    matrix_similarity_indexer = f"{DATA_DIR}/models/matrix_similarity.model"
    stopwords_filename = f"{DATA_DIR}/models/vietnamese-stopwords-dash.txt"
    stopwords = load_stopwords()
    ic(len(stopwords))
    texts = load_data(filename)
    ic(texts[0:2])
    phrase_model = Phrases(texts,
                           min_count=5,
                           threshold=1,
                           common_terms=stopwords)
    # Export the trained model = use less RAM, faster processing. Model updates no longer possible.
    # phrase_model = phrase_model.freeze()
    phrase_model.save(phrase_model_name)
    texts = phrase_model[texts]
    for t in texts[0:2]:
        ic(t)
    dictionary = corpora.Dictionary(texts)
    dictionary.save(dictionary_filename)
    ic(len(dictionary))
    corpus = [dictionary.doc2bow(text) for text in texts]
    ic("Train LSI Model")
    lsi_model = LsiModel(corpus, id2word=dictionary, num_topics=100)
    lsi_model.save(model_name)

    index = MatrixSimilarity(lsi_model[corpus])  # transform corpus to LSI space and index it
    doc = "Thảm nhào bột silicon to"
    doc = phrase_model[utils.clean_text(doc).lower().split()]
    vec_bow = dictionary.doc2bow(doc)
    vec_lsi = lsi_model[vec_bow]  # convert the query to LSI space
    # ic(vec_lsi)
    ic("Save Matrix Similarity")
    index.save(matrix_similarity_indexer)
    index = MatrixSimilarity.load(matrix_similarity_indexer)
    sims = index[vec_lsi]  # perform a similarity query against the corpus
    # ic(list(enumerate(sims[:10])))  # print (document_number, document_similarity) 2-tuples
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    for doc_position, doc_score in sims[:10]:
        print(doc_score, texts[doc_position])
