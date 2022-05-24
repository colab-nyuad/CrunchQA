import torch
import random
import os
import unicodedata
import re
import time
import json
import pickle

from torch import nn
from collections import defaultdict
from tqdm import tqdm
import numpy as np

class CheckpointLoader():
    def __init__(self, embedding_path):
        self.embedding_path = embedding_path
    
    def load_checkpoint(self, args, embed_model):
        
        checkpoint = '{}/checkpoint_best.pt'.format(self.embedding_path)
        loaded_checkpoint = torch.load(checkpoint)
        entity2idx = loaded_checkpoint['entity_to_id_dict']
        rel2idx = loaded_checkpoint['relation_to_id_dict']
        entity_matrix = loaded_checkpoint['model_state_dict']['entity_embeddings._embeddings.weight']
        print(len(entity2idx))
        print(len(rel2idx))
        rel_matrix = loaded_checkpoint['model_state_dict']['relation_embeddings._embeddings.weight']
        num_relations = len(rel2idx)
        args.dim = len(entity_matrix[0])
        
        if hasattr(embed_model, 'embeddings'):
            embed_model.embeddings[0] = nn.Embedding.from_pretrained(entity_matrix, freeze = args.freeze)
            embed_model.embeddings[1] = nn.Embedding.from_pretrained(rel_matrix[:num_relations,:], freeze = args.freeze)
        else:
            embed_model.entity = nn.Embedding.from_pretrained(entity_matrix, freeze = args.freeze)
            embed_model.rel = nn.Embedding.from_pretrained(rel_matrix[:num_relations,:], freeze = args.freeze)

        args.sizes = (len(entity_matrix), len(rel_matrix) // 2, len(entity_matrix))

        return entity2idx, rel2idx

