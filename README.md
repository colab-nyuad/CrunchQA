# FinQA: New Challenges in Question Answering over Knowledge Graphs

![Image of Version](https://img.shields.io/badge/version-v1.0-green)
![Image of Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)
![my badge](https://badgen.net/badge/uses/pykeen/red?icon=github)

FinQA is a new dataset for question-answering on knowledge graphs (KGQA). The dataset was created to reflect the challenges we identified in real-world applications which are not covered by existing benchmarks, namely, multi-hop constraints, numeric and literal embeddings, ranking, reification, and hyper-relations. 

The repository contains scripts for:
- creating a Knowledge Graph from the Crunchbase database;
- creating a Question Answering dataset based on multiple-hop templates and paraphrasing;
- running experiments with state-of-the-art KGQA models on FinQA. 

**⚠️ IMPORTANT: Since the Crunchbase dataset is subject to licensing, the repository contains a script to process a dump and reconstruct KG. The dump provided by Cruchbase under the academic license contains all records till the current timestamp. To match the KG we used to generate questions, the script *construct_kg.py* processes data records until the given timestamp (December 2021 to match our KG)**.

*FinQA can be downloaded from [link]().*

### Quick start
```sh
# retrieve and install project in development mode
git clone https://github.com/colab-nyuad/FinQA

# set environment variables
source set_env.sh
```

## Table of contents
1. [Data](#data)
2. [KG constraction](#kg)
    1. [KG construction from CSV](#kg_csv)
    2. [KG construction from RDF](#kg_rdf)
3. [QA templates](#qa_templates)
    1. [Constraints Format](#constraints)
4. [QA dataset](#qa_dataset)
5. [Training KGQA model](#kgqa_model)
    1. [Training Embeddings](#embeddings)
    2. [Running KGQA](#kgqa)

## Data <a name="data"></a>
Download Crunchbase dump and unzip into the folder data

## Creating KG from crunchabse data  <a name="kg"></a>

The Crunchbase data dump comprises 17 relational tables with primary and foreign keys to link tables together. To build the KG, we use a simple approach. We create new entities for each main entity type and use reification nodes to map the relationship between the base entity types and link additional information like start date, end date, title, etc. For the job titles we limit the range to a set of categories that can be found in *jobs.json*.

**KG construction from CSV**  <a name="kg_csv"></a>

The knowledge graph generated from the csv dump includes 3.2 million entities, 31 relations, and 17.6 million triples. Following is the structure of the created KG:

![](kg.jpg "KG architecture")

The command to generate KG:
```python
python construct_kg.py
```

**KG construction from RDF** <a name="kg_rdf"></a>

In the paper [*A Linked Data Wrapper for CrunchBase*](http://dbis.informatik.uni-freiburg.de/content/team/faerber/papers/CrunchBaseWrapper_SWJ2017.pdf), authors proposed a wraper around the CruchBase API that provides data in RDF format. The paper includes a link to the dump dated October, 2015. Since this dump is publicly available, we map the RDF data to the KG tirples fromat and provide a link for download. The mapped KG is smaller than the used KG for constracting the questions. This version is missing events and a set of atrributes for other entities but contains the product entities. This smaller version of KG and the scheme of it can be downloaded from [link](). 

## QA templates <a name="qa_templates"></a>

The templates are classified into 3 categories: 
* 1-hop (inferential chain of the length 1 and at most 1 constraint)
* 2-hop (inferential chain of the length 2 and at most 1 constraint) 
* advanced - the rest  

Following is an example of the advanced question template:
```json
   {
    "main_chain": "org1-in_org-job1-has_job-person",
    "question": [
       "[org1] alumni who founded more than 5 companies",
       "founders of more than 5 companies who previously worked in [org1]",
       "list people who formerly worked in [org1] and founded more than 5 companies"
    ],
    "constraints": [
      {
       "entity_constraint": {
        "job1-is_current-job_current": ["False"]
       }
      },
      {
        "entity_constraint": {
          "person-has_job-job2-job_title-job_title": ["founder", "co-founder"]
         }
      },
      {
        "numeric_constraint": {
         "job2-in_org-org2": {
          "count_over":"org2",
          "group_by": ["person"],
          "numeric": ["", ">", 5]
         }
        }
       }
     ]
   }
```

Each template contains:
* main_chain - a path in the KG leading form the head entity to the answers in the format (entity<sub>1</sub>-relation<sub>1</sub>-entity<sub>2</sub>-relation<sub>2</sub>-.... relation<sub>n</sub>-entity<sub>n+1</sub>). 
* question - a language form of the question where '[]' indicates the head entity and '()' indicate contraints entitites to be replaced when questions are generated from the template
* constraints (*entity constraint*, *temporal constraint*, *maximum constraint*, *numeric constraint*)
* type: temporal when a template requires temporal data, numeric when a template requires numeric data 


The templates support multi-entity/relation type (format: entity<sub>1</sub>/entity<sub>2</sub> or relation<sub>1</sub>/relation<sub>2</sub>) to cover questions, which can refer to multiple entities or relations, e.g., if we ask about investors, both companies and people can make investments, or if a question is about participating in an event without specifying a specific role, we should encounter all types of relations, i.e., sponsor, speaker, organizer, contestant and exhibitor. Subscripts (*job<sub>1</sub>*, *job<sub>2</sub>*) refer to the same column as *job*. The order is introduced for the convience of our implementation to simplify the join while attaching the constraints. Following the table shows the number of created templates. 


|   | 1-hop  | 2-hop  | advanced  | total |
| :---        |    :----:  |    :----:  |    :----:  |    :----:  |
|FinQA     |107   |70   |66   |243    |



**Constraints Format** <a name="constraints"></a>

In the following, in the description of each constraint type "constraint_chain" indicates a constraint inferential path, which can be up to 2-hop since some constraints involve the reification nodes. The begining of the chain is one of the entities from the main inferential chain (format: entity<sub>main_chain</sub>-relation<sub>1</sub>-entity<sub>1</sub>-relation<sub>2</sub>-entity<sub>2</sub>).

***Entity constraint*** requires the tail entity of the constraint_chain to be equal to a certain value (constraint_chain: [value_<sub>1</sub>, value_<sub>2</sub>, ..., value_<sub>n</sub>].  E.g., for queries asking about female employees it can be specified as gender = 'female':
```json
    "entity_constraint": {
      "person-gender-gender": ["female"]
     }
```

***Temporal constraint*** requires the date to be within a specified time range (constraint_chain: {} or {"before":year} or {"after":year} or {"between": [year1, year2]}). If the time range is not specified, we just sample with the indicated year or year+month depending on setting granularity. E.g, for queries asking about funding rounds announced between 2010 and 2020:
```json
     "temporal_constraint": {
        "funding_round-announced_on-date": {
           "between": ["2010", "2020"]
        }
       }
```

***Maximum constraint*** is introduced to reflect key words as "top", "at most", "the highest" and etc. Maximum is always computed within a group and works in two settings: grouping by the head entity and taking maximum by the column or first counting over edges and then selecting maximum.

1. Grouping by the head entity is the default setting so we just need to specify by which column the maximum needs to be taken (constraint_chain: {"max": entity}). E.g, for the query asking which Software company raised the highest amount of money in its ipo, the question generator will group by "Software" (org_category) and select maximum among raised_price:
```json
    "max_constraint": {
     "org-org_ipo-ipo-raised_amount_usd-raised_price": {
      "max": "raised_price"
     }
    }
```
2. Another setting the maximum constraint supports is first counting over edges and then selecting maximum. For this setting we need to specify three fields:
    - "count_group_by" - the list of columns to group by
    - "count_over" - the column we count over while grouping and from wich we will select the maximum
    - "max_group_by": the entity around which the question is centered
    
E.g., for the query asking what types of events did Bill Gates mostly participate in, we need to group by Bill Gates and type of the events he participated in (person, event_type) and while grouping we count how many times (we count over the column events), after grouping we select maximun from the count for Bill Gates, in the notation of templates we select maximum from the count for central entity (person):
```json
   "max_constraint": {
     "event-type_of-event_role": {
       "count_over": "event",
       "count_group_by": ["person", "event_type"],
       "max_group_by": ["person"],
     }
   }
```
***Numeric constraint*** reflects the key words "more than", "less than", "at least". This constraint also supports two settings: filtering records by applying the specified condition on the columns values or counting over edges and then applying the specified condition on this count. For the first setting we just need to specify the condition in the format ["entity", ">|=|<",200]. For the second setting we need to specify:
- "group_by" - the list of columns to group by
- "count_over" - the column we count over while grouping and on which the condition will be applied
- "numeric": condition in the format ["", ">|=|<", number]
E.g., for a query asking to list companies which acquired more than 50 companies, we need to group by organizations and count over organizations it acquired, then filter out the records where count is <= than 50:
```json
    "numeric_constraint": {
      "org1-is_acquirer-acquisition-is_acquiree-org2": {
       "count_over":"org2",
       "group_by": ["org1"],
       "numeric": ["", ">", 50]
      }
   }
```

## QA dataset <a name="qa_dataset"></a>

To generate the dataset from the templates:
```python
python genrate_dataset.py --sample_size 200
```
Sample size indicates how many questions per template to genrate. We generetated 100 questions per 1, 2-hop templates and 200 per advanced templates. Command to shuffle and split generated questions into train, valid, test: 
```
./qa_dataset/split_train_valid_test.bh
```

## Training the model and running experiments <a name="kgqa_model"></a>

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
