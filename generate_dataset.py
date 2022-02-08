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
folders = ["qa/1hop/simple_constraint", 
           "qa/1hop/base", 
           "qa/1hop/temporal_constraint", 
           
           "qa/2hop/base", 
           "qa/2hop/simple_constraint", 
           "qa/2hop/temporal_constraint", 
           
           "qa/advanced/counting", 
           "qa/advanced/ordering", 
           "qa/advanced/multi_entity", 
           "qa/advanced/multi_relation",
           "qa/advanced/multi_constraint",
           "qa/advanced/time_range", 
           "qa/advanced/quantity_range"]

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
record_paths = ["qa/1hop/1hop_record.txt", "qa/2hop/2hop_record.txt", "qa/advanced/acvanced_record.txt"]
for record_path in record_paths:
    if os.path.exists(record_path):
        os.remove(record_path)
        print("successfully deleted old" + record_path)
    else:
        print(record_path + "doesn't exist yet, will create a new one")
        
    open(record_path, 'a').close()
    print("created new" + record_path)


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
#print(cluster_centers)

#*************************************for checking***************************************
#print(country_code_to_word["CAN"])
#print(country_code_to_word["ROM"])
#print(country_code_to_word["BAH"])


#*********************************************************************************
#*************************************1 hop QA************************************
#*********************************************************************************
'''
t1 = pd.read_csv(temp_path + "template_1hop.csv")
print(t1)


# ----------------------base questions----------------------
t1_base = t1[["entity1", "relation1", "entity2", "1_hop_q"]]
t1_base = t1_base.dropna()
t1_base = t1_base.drop_duplicates()
for index, row in t1_base.iterrows():
    head = row['entity1']
    rel = row['relation1']
    tail = row['entity2']
    print(head, rel, tail)
    
    entity_route = head + "-" + rel + "-" + tail
    
    # for base questions
    question = row["1_hop_q"]
    qa_filename = "qa-1hop-" + str(index) + ".txt"
    
    qa = pick_answer_1hop(temp_path, triples_path, head, rel, tail, question, sample_size, qa_filename)
    write_df_qa(qa, "qa/1hop/base/", qa_filename, question)
    write_qa_record("qa/1hop/1hop_record.txt", entity_route, question, index, len(qa), "None")
    
    


#------------------------simple constraint---------------
t1_simple = t1[["entity1", "relation1", "entity2", "simple_constraint", "1_hop_q_simple_constraint"]]
t1_simple = t1_simple.drop_duplicates()
t1_simple = t1_simple.dropna()

#print(t1_simple.simple_constraint.to_list())


for index, row in t1_simple.iterrows():
    head = row['entity1']
    rel = row['relation1']
    tail = row['entity2']
    print(head, rel, tail)
    
    entity_route = head + "-" + rel + "-" + tail
    simple_constraint = row["simple_constraint"]
    simple_question = row["1_hop_q_simple_constraint"]
    qa_filename = "qa-1hop-" + str(index) + "simple.txt"
    qa = pick_answer_1hop_constraint(temp_path, triples_path, head, rel, tail, simple_question, sample_size, qa_filename, constraint = simple_constraint)
    write_df_qa(qa, "qa/1hop/simple_constraint/", qa_filename, simple_question)
    write_qa_record("qa/1hop/qa_record.txt", entity_route, simple_question, index, len(qa), simple_constraint)



#-------------------------temporal questions------------------------
t1_temporal = t1[["entity1", "relation1", "entity2", "temporal_constraint", "1_hop_1_temporal_constraint"]]
t1_temporal = t1_temporal.drop_duplicates()
t1_temporal = t1_temporal.dropna()

for index, row in t1_temporal.iterrows():
    head = row['entity1']
    rel = row['relation1']
    tail = row['entity2']
    print(head, rel, tail)
    entity_route = head + "-" + rel + "-" + tail
    temporal_constraint = row["temporal_constraint"]
    temporal_question = row["1_hop_1_temporal_constraint"]
    qa_filename = "qa-1hop-" + str(index) + "temporal.txt"
    qa = pick_answer_1hop_temporal(temp_path, triples_path, head, rel, tail, temporal_question, sample_size, qa_filename, constraint = temporal_constraint)
    write_df_qa(qa, "qa/1hop/temporal_constraint/", qa_filename, temporal_question)
    write_qa_record("qa/1hop/1hop_record.txt", entity_route, temporal_question, index, len(qa), temporal_constraint)
    qa = pick_answer_1hop_temporal(temp_path, triples_path, head, rel, tail, temporal_question, sample_size, qa_filename, constraint = temporal_constraint, year_only = True)
    write_df_qa(qa, "qa/1hop/temporal_constraint/", "qa-1hop-" + str(index) + "temporal-year_only.txt", temporal_question)
    write_qa_record("qa/1hop/1hop_record.txt", entity_route, temporal_question + "@year_only", index, len(qa), temporal_constraint)

        
print("*"*50)
'''


#*********************************************************************************
#*************************************2 hop QA************************************
#*********************************************************************************
'''
t2 = pd.read_csv(temp_path + "template_2hop.csv")
print("this is t2")
print(t2)


#--------------------------------------base questions------------------------
t2_base = t2[["entity1", "relation1", "entity2", "relation2", "entity3", "2_hop_q"]]
t2_base = t2_base.drop_duplicates()
t2_base = t2_base.dropna()

for index, row in t2_base.iterrows():
    
    ent1 = row['entity1']
    rel1 = row['relation1']
    ent2 = row['entity2']
    rel2 = row["relation2"]
    ent3 = row["entity3"]
    components = [ent1, rel1, ent2, rel2, ent3]
    print("components:", components)
    
    entity_route ="-".join(components)
    
    # for ordinary questions
    question = row["2_hop_q"]
    qa_filename = "qa-2hop-" + str(index) + ".txt"
    qa = pick_answer_2hop(temp_path, triples_path, ent1, rel1, ent2, rel2, ent3, question, sample_size, qa_filename)
    write_df_qa(qa, "qa/2hop/base/", qa_filename, question)
    write_qa_record("qa/2hop/2hop_record.txt", entity_route, question, index, len(qa), "None")
'''

'''
#-------------------------------simple constraint---------------------

t2_simple = t2[["entity1", "relation1", "entity2", "relation2", "entity3", "2_hop_q", "simple_constraint", "2_hop_1_simple_constraint"]]
t2_simple = t2_simple.drop_duplicates()
t2_simple = t2_simple.dropna()

for index, row in t2_simple.iterrows():
    
    ent1 = row['entity1']
    rel1 = row['relation1']
    ent2 = row['entity2']
    rel2 = row["relation2"]
    ent3 = row["entity3"]
    simple_constraint = row["simple_constraint"]
    components = [ent1, rel1, ent2, rel2, ent3]
    print("components:", components)
    
    entity_route ="-".join(components)
    
    # for ordinary questions
    question = row["2_hop_1_simple_constraint"]
    qa_filename = "qa-2hop-" + str(index) + "simple.txt"
    qa = pick_answer_2hop_constraint(triples_path, ent1, rel1, ent2, rel2, ent3, question, sample_size, simple_constraint)
    write_df_qa(qa, "qa/2hop/simple_constraint/", qa_filename, question)
    write_qa_record("qa/2hop/2hop_record.txt", entity_route, question, index, len(qa), simple_constraint)

'''

#-------------------------------temporal constraint-------------------------


#*********************************************************************************
#*************************************2 hop QA************************************
#*********************************************************************************

ta = pd.read_csv("qa_templates/template_advanced.csv")
print(ta)


# to do
# (done) add " ".join for category and category group
# [event_rold] event delete event in sentence
# for clustered, add pickle value
# write for ordinal constraint
# fix grammar errors
# put people land organization together (in advanced qa)







    