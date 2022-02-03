# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:17:32 2022

@author: yulif
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 21:24:32 2022

@author: yulif
"""

# to do:
    # 1. write an if statement for [year]
    # 2. special case for acquisition type, not select "acquisition"
    # 3. remember to drop duplicates of 1 hop questions
    # (done) 4. write functions for "readables":
        # country 
        # company 
        # person
        # event
        # funding round
        # fund
    # 5. special case for gender, sample from each gender
    # 6. before sampling any qa, check if the dataframe length is greater than 0
    # 7. for date, when date1 and date2 exist, write if statement
    
    
import pandas as pd
import random
import pickle
import os
from utils_dataset import *

# clear previous contents
folders = ["qa/1hop/csv/", "qa/1hop/txt/", "qa/2hop/csv", "qa/2hop/txt", "qa/advanced/csv", "qa/advanced/txt"]
for folder in folders:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
            
# remove and create file that record the question dataset size
record_path = "qa_templates/qa_dataset_overview.txt"
if os.path.exists(record_path):
    os.remove(record_path)
    print("successfully deleted old qa_dataset_overview.txt")
else:
    print("qa_dataset_overview.txt doesn't exist yet, will create a new one")
    
open(record_path, 'a').close()
print("created new qa_dataset_overvies.txt")


# path for the triplets
triples_path = "triples/"

# path for files that convert non-readables to readables
readable_path = "qa_readable/"

# path for templates
temp_path = "qa_templates/"

# this many questions per template
sample_size = 100


cluster_file = open("clustering/clusters.pickle", "rb")
cluster_centers = pickle.load(cluster_file)
cluster_file.close()
print(cluster_centers)

#*************************************for checking***************************************
print(country_code_to_word["CAN"])
print(country_code_to_word["ROM"])
print(country_code_to_word["BAH"])


#**************************************config*******************************
config = {
    "1hop":[
        [None, '1_hop_q'],
        ['simple_constraint', '1_hop_q_simple_constraint']
        
        
        
        
        ]
    
    
    }


#*********************************************************************************
#*************************************1 hop QA************************************
#*********************************************************************************

t1 = pd.read_csv(temp_path + "template_1hop.csv")
print(t1)

# constraints and questions to read
# [constraint, question]

for index, row in t1.iterrows():
    head = row['entity1']
    rel = row['relation1']
    tail = row['entity2']
    question = row["1_hop_q"]
    qa_filename = "qa" + str(index)
    print(head, rel, tail)
    triplet_df = get_df(triples_path, head, rel, tail)
    print(len(triplet_df))
    
    pick_answer_1hop(temp_path, triples_path, head, rel, tail, question, sample_size, qa_filename)
    
    print("*"*50)




    