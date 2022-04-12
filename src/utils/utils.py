#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/12/2021

@author: phongdk
"""

import logging.config
import os
import pickle
import re
import shutil
import sys
import unicodedata
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
from gensim.utils import simple_preprocess
from pyvi import ViTokenizer

from src import config
from src.utils.md5checksum import write_checksum

logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

DATA_DIR = os.getenv("DATA_DIR")

CEPH_CONNECTION_ERROR = 107
PRODUCT_CODE_REGEX = "((?:[a-z]+\S*\d+|\d\S*[a-z]+)[a-z\d_-]*)"


def clean_text(sentence):
    # Remove new line characters
    sentence = re.sub('\s+', ' ', sentence)
    # Remove distracting single quotes
    sentence = re.sub("\'", "", sentence)
    # Remove distracting double quotes
    sentence = re.sub("\"", "", sentence)
    # '[^\w\d]',  # split on non-words and non-digits (eliminates punctuation)
    # Remove text between [], e.g [Free ship]
    sentence = re.sub(r'\[(.+?)\]', '', sentence)
    return sentence.strip()


def normalize_NFC(text: str):
    return unicodedata.normalize('NFC', text)


def tokenize(text: str):
    return ViTokenizer.tokenize(normalize_NFC(text))


def process_text(text: str):
    return simple_preprocess(tokenize(text))


def normalize_domain(domain):
    if domain:
        domain = str(domain).replace("www.", "")
        if domain.startswith("m."):
            domain = domain.replace("m.", '')
    return domain


def normalize_directory(directory, max_words=5, min_length=2, max_length=30):
    if directory:
        directory = str(directory).lower().strip()
        return directory if len(directory.split("-")) <= max_words and min_length < len(directory) < max_length \
            else None
    return directory


def extract_domain(url):
    try:
        url = str(url)
        x = urlparse(url).netloc
        return x.strip()
    except:
        return None


def extract_directory(url):
    url_parse = urlparse(url)
    directory = url_parse.path.split("/")
    if len(directory) > 2:
        if directory[1] not in ["vn", "vi", "vi-vn", "vietnamese"]:
            directory = directory[1]
        else:
            directory = directory[2]
    else:
        directory = None
    return directory


def unix_time_to_time(unix_time):
    # if you encounter a "year is out of range" error the timestamp
    # may be in milliseconds, try `ts /= 1000` in that case
    return datetime.utcfromtimestamp(int(unix_time)).strftime('%Y-%m-%d %H:%M:%S')


def get_top_elements_by_abs_percentage(elements, max_elements=3, abs_percentage=0.1):
    """
    :param elements: list of score
    :param max_elements:
    :param abs_percentage:
    :return:
    """
    sorted_idx = np.argsort(elements)[::-1]
    selected_idx = [sorted_idx[0]]
    for idx in sorted_idx[1:]:
        if elements[sorted_idx[0]] - elements[idx] < abs_percentage * elements[sorted_idx[0]]:
            selected_idx.append(idx)
            if len(selected_idx) == max_elements:
                break
    return selected_idx


def get_top_elements(aggregate_reading, num_top):
    """
    :param df_article: dataframe of article
    :param available_articles:
    :param reading_weight: weight of reading articles, None if don't care time reading
    :param column:
    :param num_top: number of top category/topic to recommend
    :return:
    """
    top_elements_idx = np.argsort(aggregate_reading)[::-1][:num_top].tolist()
    top_elements_weight = aggregate_reading[top_elements_idx]
    top_elements_weight = list((top_elements_weight / top_elements_weight[0]).astype(np.float16))
    return top_elements_idx, top_elements_weight


def preprocess(text):
    tokens = ViTokenizer.tokenize(text.lower())
    tokens = tokens.split(',')
    new_tokens = [element.strip().replace(' ', '_') for element in tokens]
    return new_tokens


def save_to_pickle(data, filename, is_checksum=False):
    logger.info(f'-------- Save file to : {filename} -------')
    tmp_filename = f"{filename}.tmp"
    pickle.dump(data, open(tmp_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    # rename files
    shutil.move(tmp_filename, filename)
    # write checksum
    if is_checksum:
        write_checksum(filename, f'{filename}.checksum')
    logger.info(f'-------- Done save file: {filename} -------------')


def check_ceph_connection(error):
    if hasattr(error, 'errno'):
        if int(error.errno) == CEPH_CONNECTION_ERROR:
            sys.exit()


def remove_duplicate(seq: list) -> list:
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def detect_branch(text: str):
    pass


def detect_product_code(text: str):
    list_codes = re.findall(pattern=PRODUCT_CODE_REGEX,
                            string=text.lower())
    return list_codes
