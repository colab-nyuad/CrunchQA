# <center> FinQA</center> 
### New Challenges in Question Answering over Knowledge Graphs

![Image of Version](https://img.shields.io/badge/version-v1.0-green)
![Image of Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)
![my badge](https://badgen.net/badge/uses/pykeen/red?icon=github)

FinQA is a new dataset for question-answering on knowledge graphs (KGQA) to reflect the challenges we identified in real-world applications which are not covered by existing benchmarks, namely, multi-hop constraints, numeric and literal embeddings, ranking, reification, and hyper-relations. 

The repository contains scripts for:
- creating a Knowledge Graph from the Crunchbase database;
- creating a Question Answering dataset based on multiple-hop templates and paraphrasing;
- running experiments with state-of-the-art KGQA models on FinQA. 

**⚠️ IMPORTANT: Make sure that you have torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 exactly. You can check with**

### Quick start
```sh
# retrieve and install project in development mode
git clone https://github.com/colab-nyuad/FinQA

# set environment variables
source set_env.sh
```

### Table of contents

# Table of contents
1. [Data](#data)
2. [KG constraction] (#kg)
    1. [KG construction from CSV](#kg_csv)
    2. [KG construction from RDF](#kg_rdf)
3. [QA templates] (#qa_templates)
4. [QA dataset](#qa_dataset)
5. [Training KGQA model](#kgqa_model)
    1. [Training Embeddings](#embeddings)
    2. [Running KGQA](#kgqa)

### Data <a name="data"></a>
Download Crunchbase dump and unzip into the folder data

### Creating KG from crunchabse data 

The Crunchbase data dump comprises 17 relational tables\footnote{The database dump was obtained on December 2021.} with primary and foreign keys to link tables together. To build the KG, we use a simple approach. First, we create new entities for each main entity type, namely, \textsc{Organization, Person, Fund, Event}, and \textsc{Funding Round}. 

Next, we use reification nodes \textsc{Job}, \textsc{Acquisition} and \textsc{IPO} to map the relationship between the base entity types. For example, instances of \textsc{Job} are entities that link instances of \textsc{Organization, Person} and add additional information about this particular position, for example, start date, end date, title, etc. 

Finally, we map additional triples extracted from the entities' respective tables including numerical values and dates. At the same time, we exclude fields with textual literals and various metadata irrelevant to our task. The processed knowledge graph includes 3.2 million entities, 31 relations, and 17.6 million triples. 

Figure \ref{fig:kg_architecture} shows the overall structure of our Knowledge graph, where rectangular nodes denote the main classes of entities and the circular nodes visualize the reification we perform. A detailed description of all the entities and relation types is available on the dataset's website. In addition, since the Crunchbase dataset is subject to licensing, we provide a script to process a dump and reconstruct the KG at a given timestamp.


Mapped Jobs:

| KG format   | CSV format  |
| :---        |    :----:   |
|director     |director    |
|CSO          |chief security officer, cso|
|board_member |chairman, boar director, board member|
|founder      |founding partner, founder, co-founder, co founder, cofounder|
|CFO          |chief financial officer, cfo|
|CEO          |chief executive officer, ceo| 
|CPO          |chief people officer, cpo|
|CIO          |chief information officer, cio|
|CBO          |chief business officer, cbo|
|VP           |vice president, vp| 
|advisor      |advisor|
|owner        |owner|
|president    |president|
|CTO          |chief technology offcier, cto|
|CRO          |chief revenue officer, cro| 
|COO          |chief operating officer, coo|

![](kg.jpg "KG architecture")

Command to generate KG: python construct_kg.py

### QA templates 

### QA dataset


Command to generate QA datset: python genrate_dataset.py

Command to shuffle and split generated questions into train, valid, test: ./qa_dataset/split_train_valid_test.bh

### Training the model and running experiments

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

### Avilable models
This implementation includes the following models:
- [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf)
- [TransE](https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)
- [DistMult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ICLR2015_updated.pdf)


Command to shuffle and split generated triples into train, valid, test: ./kg/split_train_valid_test.bh

Command to train kg embeddings using pykeen:
python train_embeddings.py --model DistMult --train_path kg/clustering/train.txt --valid_path kg/vanilla/valid.txt --test_path kg/vanilla/test.txt --dim 200 --results_folder embeddings/clustering_distmult --gpu 1

Command to run KGQA framework: 
python run.py --model DistMult --embeddings_folder embeddings/clustering_distmult --freeze True
