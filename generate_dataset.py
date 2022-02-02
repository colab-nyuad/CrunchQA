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
from utils_dataset import *

# path for the triplets
path = "triples/"

# path for files that convert non-readables to readables
readable_path = "qa_readable/"

# path for templates
temp_path = "qa_templates/"

#*********************************investment type**********************************

investment_type_to_word = get_investment_type(readable_path)

#**********************************stock exchange symbol***********************

stock_to_word = get_stock_symbol(readable_path)

#***********************************country code****************************

country_code_to_word = get_country_dict(readable_path)

#*******************************cluster name to value********************************

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
    print(head, rel, tail)
    triplet_df = get_df(path, head, rel, tail)
    if len(triplet_df) == 0:
        print("-------------------------------------")
    print(triplet_df)
    print("*"*50)



    
    