import os
import argparse
import yaml
from shutil import copyfile
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from tqdm import tqdm
import argparse
import operator
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import networkx as nx
import ast
import json

from kgqa import *
from kgqa import emb_model, optimizer, models
from utils.utils_kgqa import *

parser = argparse.ArgumentParser(
    description="Graph Embedding for Question Answering over Knowledge Graphs"
)

parser.add_argument(
    "--model", default="ComplEx", choices=models, help="Embedding model"
)

parser.add_argument(
    "--embeddings_folder", type=str, help="Embedding path"
)

parser.add_argument(
    "--regularizer", choices=["F2"], default="F2", help="Regularizer"
)

parser.add_argument(
    "--reg", default=0.0, type=float, help="Regularization weight"
)

parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam"], default="Adam", help="Optimizer"
)

parser.add_argument(
    "--max_epochs", default=100, type=int, help = "Maximum number of epochs to train"
)

parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)

parser.add_argument(
    "--valid_every", default=3, type=int, help="Number of epochs before validation"
)

parser.add_argument(
    "--batch_size", default=128, type=int, help="Batch size"
)

parser.add_argument(
    "--learning_rate", default=0.0005, type=float, help="Learning rate for KGQA"
)

parser.add_argument(
    "--freeze", default=False, type=bool, help="Freeze weights of trained embeddings"
)

parser.add_argument(
    '--use_cuda', type=bool, default=True, help = "Use gpu"
)

parser.add_argument(
    '--gpu', type=int, default=0, help = "Which gpu to use"
)

parser.add_argument(
    '--num_workers', type=int, default=15, help=""
)

parser.add_argument(
    '--labels_smoothing', type=float, default=0.0, help = "Perform label smoothing"
)

parser.add_argument(
    '--decay', type=float, default=1.0
)

# Exporting enviromental variables

qa_dataset_path = os.environ['QA_DATASET_PATH']
embeddings_path = os.environ['EMBEDDINGS']
kgqa_checkpoints = os.environ['CHECKPOINTS']
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#-------------------------------------

def train(optimizer, model, data_loader, scheduler, train_samples, valid_samples, test_samples, args, checkpoint_path):

    best_score = -float("inf")
    no_update = 0
    eps = 0.0001

    phases = ['train'] * args.valid_every
    phases.append('valid')

    for epoch in range(args.max_epochs):
        for phase in phases:

            if phase == 'train':
                model.train()
                loader = tqdm(data_loader, total=len(data_loader), unit="batches")
                score = optimizer.train(loader, epoch)
                scheduler.step()

            elif phase=='valid':
                model.eval()
                score, _ = optimizer.calculate_valid_loss(train_samples)
                print('Training accuracy : ', score)
                score, _ = optimizer.calculate_valid_loss(valid_samples)

                if score > best_score + eps and epoch < args.max_epochs:
                    best_score = score
                    no_update = 0

                    print("Validation accuracy increased from previous epoch", score)
                    test_score, _ = optimizer.calculate_valid_loss(test_samples)
                    print('Test score for best valid so far:', test_score)
                    torch.save(model, '{}'.format(checkpoint_path))
                    print('Model saved')

                elif (score < best_score + eps) and (no_update < args.patience):
                    no_update +=1
                    print("Validation accuracy decreases to %f from %f, %d more epoch to check"%(score, best_score, args.patience-no_update))

                if no_update == args.patience or epoch == args.max_epochs-1:
                    print("Model has exceed patience or reached maximum epochs")
                    return

if __name__ == "__main__":
    args = parser.parse_args()

    ## Reading QA dataset
    train_samples, valid_samples, test_samples = read_qa_dataset(qa_dataset_path)

    ## Loading trained pretrained KG embeddings
    embedding_path = ('{}/{}'.format(embeddings_path, args.embeddings_folder))
    loader = CheckpointLoader(embedding_path)
    device = torch.device(args.gpu if args.use_cuda else "cpu")
    embed_model = getattr(emb_model, args.model)(args)
    entity2idx, rel2idx = loader.load_checkpoint(args, embed_model)

    ## Create QA dataset 
    print('Process QA dataset')
    word2idx,idx2word, max_len = get_vocab(train_samples)
    vocab_size = len(word2idx)
    dataset = Dataset_SBERT(train_samples, word2idx, entity2idx)
    data_loader = DataLoader_SBERT(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    
    ## Creat QA model
    print('Creating QA model')
    qa_model = SBERT_QAmodel(args, embed_model, vocab_size)
    qa_model.to(device)

    ## Create QA optimizer
    print('Creating QA optimizer')
    optimizer = getattr(torch.optim, args.optimizer)(qa_model.parameters(), lr=args.learning_rate)
    scheduler = ExponentialLR(optimizer, args.decay)
    qa_optimizer = QAOptimizer(args, qa_model, optimizer, dataset, device)

    ## Train the model
    checkpoint_path =  "{}/{}.pt".format(embedding_path, args.model)
    train(qa_optimizer, qa_model, data_loader, scheduler, train_samples, valid_samples, test_samples, args, checkpoint_path)
