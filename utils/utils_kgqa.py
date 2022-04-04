import torch, yaml, os
from shutil import copyfile
import networkx as nx
import numpy as np
from tqdm import tqdm
import pickle
from numpy import linalg as LA
import re

def get_vocab(data):
    word2idx = {}
    idx2word = {}
    for d in data:
        sent = d[1]
        for word in sent.split():
            if word not in word2idx:
                idx2word[len(word2idx)] = word
                word2idx[word] = len(word2idx)
    return (word2idx, idx2word, len(word2idx))

def process_text_file(text_file):
    data_array = []
    heads = []
    with open(text_file, 'r') as data_file:
        for data_line in data_file.readlines():
            data_line = data_line.strip().split('\t')
            question = data_line[0]
            head = data_line[1]
            ans = data_line[2].split(' || ')
            data_array.append([head, question.strip(), ans])

        return data_array


def read_qa_dataset(dataset_path):
    train_data = process_text_file('{}/train.txt'.format(dataset_path))
    test_data = process_text_file('{}/test.txt'.format(dataset_path))
    valid_data = process_text_file('{}/valid.txt'.format(dataset_path))
    return (train_data, valid_data, test_data)

def read_kg_triplets(dataset, type):
    file = '{}/{}.txt'.format(dataset, type)
    triplets = []
    with open(file, 'r') as data_file:
        for data_line in data_file.readlines():
            data = data_line.strip().split('\t')
            triplets.append(data)
        return np.array(triplets)

def read_dict(dict_):
    with open(dict_) as f:
        dictionary_ = {int(k): v for line in f for (k, v) in [line.strip().split(None, 1)]}
    return dictionary_

















