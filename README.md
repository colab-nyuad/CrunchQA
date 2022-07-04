# <center>FinQA</center> 
### New Challenges in Question Answering over Knowledge Graphs
The digital transformation in the finance sector has been driven by the advances made in big data and artificial intelligence technologies. 
For instance, data integration enables businesses to make better decisions by consolidating and mining heterogeneous data repositories.
In particular, knowledge graphs (KGs) are used to facilitate the integration of disparate data sources and can be utilized to answer complex natural language queries.
This work proposes a new dataset for question-answering on knowledge graphs (KGQA) to reflect the challenges we identified in real-world applications which are not covered by existing benchmarks, namely, multi-hop constraints, numeric and literal embeddings, ranking, reification, and hyper-relations.
To build the dataset, we create a new Knowledge Graph from the Crunchbase database using a lightweight schema to support high-quality entity embeddings in large graphs. Next, we create a Question Answering dataset based on multiple-hop templates and paraphrasing.
Finally, we conduct extensive experiments with state-of-the-art KGQA models and compare their performance on FinQA. The results show that the existing models do not perform well, for example, on multi-hop constrained queries. Hence, FinQA can be used as a challenging benchmark dataset for future KGQA reasoning models. The dataset and scripts are available on the project repository.

### Quick start
```sh
# retrieve and install project in development mode
git clone https://github.com/colab-nyuad/FinQA

# set environment variables
source set_env.sh
```


### Data
Download Crunchbase dump and unzip into folder data

### Creating KG from crunchabse data
Command to generate KG: python construct_kg.py

![](kg.jpg "KG architecture")


### Avilable models
This implementation includes the following models:
- [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf)
- [RotataE](https://arxiv.org/pdf/1902.10197.pdf)
- [TransE](https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)
- [Simple](https://arxiv.org/pdf/1802.04868.pdf)
- [CP](https://arxiv.org/pdf/1806.07297.pdf)
- [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf)
- [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf)
- [TuckER](https://arxiv.org/pdf/1901.09590.pdf)
- [RotH](https://aclanthology.org/2020.acl-main.617.pdf)
- [RefH](https://aclanthology.org/2020.acl-main.617.pdf)
- [AttH](https://aclanthology.org/2020.acl-main.617.pdf)

### Datasets
The repo presents results for two QA datasets MetaQA and WebQuestionsSP. For description of the underlying KGs please refer to the baseline paper for more details [Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings](https://www.aclweb.org/anthology/2020.acl-main.412/). The datasets are availbale for download [here](https://drive.google.com/file/d/1_hAbc5EJX3N1wWs1lo--XJUS-QYv9g3M/view?usp=sharing). Unzip KGs datasets into kge/data and QA datasets into the folder data/QA_data.

### Usage
To train and evaluate a QA task over KG, use the main.py script:

```sh
usage: main.py [-h] [--dataset DATASET] [--kg_type KG_TYPE]
              [--model {TransE,RESCAL,CP,Distmult,SimplE,RotH,RefH,AttH,ComplEx,RotatE}]
              [--hops HOPS] [--regularizer {L3}] 
              [--reg REG] [--optimizer {Adagrad,Adam}]
              [--max_epochs MAX_EPOCHS] [--valid_every VALID]
              [--dim RANK] [--patience PATIENCE]
              [--batch_size BATCH_SIZE]
              [--learning_rate LEARNING_RATE]
              [--freeze FREEZE] [--use_cuda USE_CUDA]
              [--num_workers NUM_WORKERS]
              [--qa_nn_type {LSTM,RoBERTa}] ---gpu GPU]
              [--use_relation_matching USE_RELATION_MATCHING]
              [--labels_smoothing LABELS_SMOOTHING]
 
Knowledge Graph QA

arguments:
  -h, --help            show this help message and exit
  --dataset             Knowledge Graph dataset
  --kg_type             Type of graph (full, sparse)
  --model {TransE,RESCAL,CP,Distmult,SimplE,RotH,RefH,AttH,ComplEx,RotatE}
                        Knowledge Graph embedding model and QA score function
  --regularizer {L3}
                        Regularizer
  --reg                 Regularization weight
  --optimizer {Adagrad,Adam}
                        Optimizer
  --max_epochs
                        Maximum number of epochs
  --patience            Number of epochs before early stopping for KG embeddings
  --valid_every         Number of epochs before validation for QA task
  --dim                 Embedding dimension
  --batch_size          Batch size for QA task 
  --learning_rate       Learning rate for QA task
  --hops                Number of edges to reason over to reach the answer 
  --freeze              Freeze weights of trained KG embeddings
  --use_cuda            Use gpu
  --gpu                 How many gpus to use
  --num_workers         Number of workers for parallel computing 
  --labels_smoothing    Labels smoothing
  --qa_nn_type {LSTM,RoBERTa}
                        Which NN to use for question embeddings
  --use_relation_matching 
                        Use relation matching for postprocessing candidates in QA task
```

Running the script main.py computes KG embeddings using [LibKGE](https://github.com/uma-pi1/kge) and QA task over KG. To compute the embeddings using LibKGE, training parameters (learning_rate, batch_size, optimizer_type, dropout, normalization_metric and etc.) need to be specified in a config file. The script checks if there is an uploaded config file in the fomrat: \<dataset\>\_\<kg_type\>\_\<model\>\_\<dim\> in the folder kge/data/config_files/<dataset> to use for training embeddings. If the file not found, the config will be created from the input arguments. 

### Sample Commands
Following is an example command to run tarining KG embedding and QA task for sparse MetaQA dataset, dimension 200, AttH model and 1hop questions: 

```sh
python main.py --dataset MetaQA --model AttH --dim 400 --kg_type half --valid_every 5 --max_epochs 200 \
--learning_rate_kgqa 0.0002 --hops 1 --qa_nn_type LSTM
```

For Freebase:
```sh
python main.py --dataset fbwq --model ComplEx --dim 50 --kg_type full --valid_every 10 --max_epochs 200 \
--learning_rate_kgqa 0.00002 --freeze True --batch_size 16 --qa_nn_type RoBERTa
```
  
To use already pretrained embeddings, please specifiy the path to the folder with files checkpoint_best.pt, entity_ids.del and relation_ids.del:

```sh
python main.py --dataset MetaQA --embeddings data/pretrained_models/MetaQA/AttH_MetaQA_half_400/ \
--model AttH --dim 400 --kg_type half --valid_every 5 --max_epochs 200 --learning_rate_kgqa 0.0002 --hops 3 \
--qa_nn_type LSTM
```
  
Compute performance with relation matching:
```sh
python main.py --dataset fbwq --model RefH --dim 400 --kg_type half --valid_every 10 --max_epochs 200 \
--learning_rate_kgqa 0.00002 --freeze True --batch_size 16 --qa_nn_type RoBERTa --use_relation_matching True
```
  

### Using pretrained models

In the following example, we load a saved kgqa checkpoint, evaluate its performance and print some questions from the dataset to explore the predicted answers.

```python
import argparse
import numpy as np
from utils.utils import *
import dataloaders, qamodels
from optimizers import QAOptimizer

# Define parameters
args = argparse.ArgumentParser().parse_args()
args.hops = 2 # 0 for fbwq
args.use_relation_matching = False
args.qa_nn_type = 'LSTM'
args.max_epochs = 100
args.batch_size = 100
n_q = 10

# Specify paths to the file entity_ids and QA dataset
entity_ids_path = 'kge/data/MetaQA_half/entity_ids.del'
qa_dataset_path = 'data/QA_data/MetaQA'

idx2entity = read_dict(entity_ids_path)
entity2idx = read_inv_dict(entity_ids_path)

# Load QA data
train_data, _, test_data = read_qa_dataset(args.hops, qa_dataset_path)
train_samples, _ = process_text_file(train_data)
test_samples, _ = process_text_file(test_data)

## Create QA dataset
word2idx,idx2word, max_len = get_vocab(train_samples)
vocab_size = len(word2idx)
dataset = getattr(dataloaders, 'Dataset_{}'.format(args.qa_nn_type))(train_samples, word2idx, entity2idx)

## Creat QA model
device = torch.device(0)
checkpoint_path = 'checkpoints/MetaQA_half_AttH_200_2.pt'
qa_model = torch.load(checkpoint_path)

## Create QA optimizer
qa_optimizer = QAOptimizer(args, qa_model, None, None, dataset, device)

score, predicted_answers = qa_optimizer.calculate_valid_loss(test_samples)
print('test score' , score)

## Print first n_q questions & predcited answers
for i, t in enumerate(test_samples[:n_q]):
    question = t[1].replace('NE', t[0])
    print('Question: {} \n Predicted Answer: {} \n Answers: {} \n'.format(question, idx2entity[predicted_answers[i]], ','.join(t[2])))
```


### How to cite



Download Crunchbase dump and unzip into folder data

Command to generate KG: python construct_kg.py

Command to shuffle and split generated triples into train, valid, test: ./kg/split_train_valid_test.bh

Command to train kg embeddings using pykeen:
python train_embeddings.py --model DistMult --train_path kg/clustering/train.txt --valid_path kg/vanilla/valid.txt --test_path kg/vanilla/test.txt --dim 200 --results_folder embeddings/clustering_distmult --gpu 1

Command to generate QA datset: python genrate_dataset.py

Command to shuffle and split generated questions into train, valid, test: ./qa_dataset/split_train_valid_test.bh

Command to run KGQA framework: 
python run.py --model DistMult --embeddings_folder embeddings/clustering_distmult --freeze True


Mapped Jobs:

"director": ["director"]
"CSO": ["chief security officer", "cso"] 
"board_member": ["chairman", "boar director", "board member"] 
"founder": ["founding partner", "founder", "co-founder", "co founder", "cofounder"] 
"CFO": ["chief financial officer", "cfo"] 
"CEO": ["chief executive officer", "ceo"] 
"CPO": ["chief people officer", "cpo"]
"CIO": ["chief information officer", "cio"] 
"CBO": ["chief business officer", "cbo"]
"VP": ["vice president", "vp"] 
"advisor": ["advisor"]
"owner": ["owner"]
"president": ["president"]
"CTO": ["chief technology offcier", "cto"]
"CRO": ["chief revenue officer", "cro"]
"COO": ["chief operating officer", "coo"]

# YOLO Task

![Image of Version](https://img.shields.io/badge/version-v1.0-green)
![Image of Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

![my badge](https://badgen.net/badge/uses/YOLOv5/red?icon=github)

Task for changing the bounding / anchor boxes from rectangles to ellipses (in train, val and detect) by Silvey Yu

**Demo:** [On YouTube](https://www.youtube.com/watch?v=unRMQn6KwF0&ab_channel=SilveyYu)

### How to run

1. Setup the virtual envrionment on your computer

```
pip3 install virtualenv
virtualenv test
source test/bin/activate
```
2. Clone this repository
```
cd test
git clone https://github.com/SilvesterYu/YOLO_Silvey_Task.git
```

3. Run setup
```
chmod -R 777 *.*
cd YOLO_Silvey_Task
chmod 0755 setup.sh
./setup.sh
```

**⚠️ IMPORTANT: Make sure that you have torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 exactly. You can check with**
```
pip3 list
```

## Testing

If "Descriptors cannot be created directly" error occurs, run:

```
pip install protobuf==3.20.1
```

**1. Test Train**

To test the train, go into ``yolov5copy`` directory
```
cd yolov5copy
```

Test train with

```
python train.py --data coco128.yaml --cfg yolov5s.yaml --weights '' --batch-size 128 --epochs 1
```

**2. Test Val**

To test the val, go into ``yolov5copy`` directory
```
cd yolov5copy
```

Test val with

```
python val.py --data coco128.yaml --weights yolov5s.pt --img 640
```

**3. Test Detection**

To test the detection, go into ``yolov5copy`` directory
```
cd yolov5copy
```

Then, get an image from internet
```
wget "https://a-z-animals.com/media/2021/12/Best-farm-animals-cow.jpg"
```

Run detection using
```
