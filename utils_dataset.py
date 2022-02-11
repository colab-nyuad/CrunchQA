# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:16:49 2022

@author: yulif
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 11:13:51 2022

@author: yulif
"""

import pandas as pd
import random
import pickle

# path for the triplets
triples_path = "triples/"

# path for files that convert non-readables to readables
readable_path = "qa_readable/"

# path for templates
temp_path = "qa_templates/"

# clustered literals
'''
cluster_file = open("clustering/clusters.pickle", "rb")
cluster_centers = pickle.load(cluster_file)
cluster_file.close()
#print(cluster_centers)
'''

#***********************************************************************#
#**********************turn strings into readables**********************#
#***********************************************************************#
# for person, org, event, funding_round, fund
def readable(x):
    x = x.strip()
    x = x.split("@")[0]
    x = " ".join(x.split("_"))
    x = x.strip()
    return x

# for country_code
def country_readable(x, d):
    # x is country code
    # d is countryd
    return d.get(x, "error")

def get_country_dict(path):
    ccode = pd.read_csv(path + "country_code.csv")
    print(ccode.columns)
    ccode.columns = ['name', 'alpha-2', "country_code", 'country-code', 'iso_3166-2', 'region',
           'sub-region', 'intermediate-region', 'region-code', 'sub-region-code',
           'intermediate-region-code']
    ccode["name"] = ccode["name"].apply(lambda x: rename_country(x))
    #print(ccode)
    countryd = dict(zip(ccode.country_code.to_list(), ccode.name.to_list()))  
    countryd.update({"ROM":"Romania"})
    countryd.update({"TAN":"Tanzania"})
    countryd.update({"BAH":"Bahamas"})
    return countryd

def rename_country(x):
        if "," in x:
            return " ".join(reversed(x.split(","))).strip()
        if "of)" in x:
            return " ".join(reversed(x[:-1].split("("))).strip()
        if ("(" in x) and ("of" not in x):
            return " ".join(x.split("("))[:-1].strip()
        return x.strip()
    
def get_stock_symbol(path):
    stock_to_word = dict()
    # if changes, edit here
    f = open(path + "stock_exchange.txt", "r")
    for line in f:
        line = line.strip("\n")
        to_add = line.split("\t")
        stock_to_word.update({to_add[0]:to_add[1]})
    f.close()
    return stock_to_word
    
def get_investment_type(path):
    investment_type_to_word = dict()
    f = open(path + "investment_type.txt", "r")
    for line in f:
        line = line.strip("\n")
        to_add = line.split("\t")
        investment_type_to_word.update({to_add[0]:to_add[1]})
    f.close()
    return investment_type_to_word
            

def turn_into_readable(entity, instance):
    # turn unreadables into readables
    if (entity == "person") or (entity == "org") or (entity == "event") or (entity == "funding_round") or (entity == "fund"):
        instance = readable(instance)
    elif (entity == "country_code"):
        instance = country_code_to_word[instance]
    elif (entity == "investment_type"):
        instance = investment_type_to_word[instance]
    elif (entity == "event_role")  or (entity == "category") or (entity == "acquisition_type") or (entity == "category_group"):
        instance = " ".join(instance.split("_"))
    elif (entity == "gender"):
        if instance == "ftm":
            instance = "transgender_man"
        elif instance == "mtf":
            instance = "transgender_woman"
        instance = " ".join(instance.split("_"))
    return instance
    

'''
def test_func(h, *args):
    print(h)
    print(args)
    if len(args) > 0:
        print("l")
    else:
        print("s")
'''

#******************************dictionaries for readables*********************#

#*********************************investment type**********************************

investment_type_to_word = get_investment_type(readable_path)

#**********************************stock exchange symbol***********************

stock_to_word = get_stock_symbol(readable_path)

#***********************************country code****************************

country_code_to_word = get_country_dict(readable_path)

#*******************************cluster name to value********************************


#***************************************************************************#
#*************************get and join triplet files************************#
#***************************************************************************#

# get the triplet dataframe corresponding to the head rel and tail
# path here refers to triples path
def get_df(path, head, rel, tail):
    try:
        filename = path + head + "-" + rel + "-" + tail + ".csv"
        main_df = pd.read_csv(filename)
    except:
        filename = path + tail + "-" + rel + "-" + head + ".csv"
        main_df = pd.read_csv(filename)
        '''
        # turn unreadables into readables
    if (head == "person") or (head == "org") or (head == "event") or (head == "funding_round") or (head == "fund"):
        main_df[head] = main_df[head].apply(lambda x: readable(x))
    elif (head == "country_code"):
        main_df[head] = main_df[head].apply(lambda x: country_code_to_word[x])
    elif (head == "investment_type"):
        main_df[head] = main_df[head].apply(lambda x: investment_type_to_word[x])
    elif (head == "event_role") or (head == "category") or (head == "category_group"):
        main_df[head] = main_df[head].apply(lambda x: " ".join(x.split("_")))
        '''

    return main_df

def get_main_df(path, main_route):
    main_route_list = main_route.split("-")
    
    if len(main_route_list) == 3:
        head = main_route_list[0]
        rel = main_route_list[1]
        tail = main_route_list[2]
        result = get_df(path,head, rel, tail)
        
    elif len(main_route_list) == 5:
        ent1 = main_route_list[0]
        rel1 = main_route_list[1]
        ent2 = main_route_list[2]
        rel2 = main_route_list[3]
        ent3 = main_route_list[4]
        triples1 = get_df(path, ent1, rel1, ent2)
        triples2 = get_df(path, ent2, rel2, ent3)
        if ent1 == ent3:
            triples1.columns = [ent1+ "_a", rel, ent2]
            triples2.columns = [ent2, rel, ent3 + "_c"]
        result = pd.merge(triples1, triples2, on = ent2)
        
    elif len(main_route_list) == 7:
        ent1 = main_route_list[0]
        rel1 = main_route_list[1]
        ent2 = main_route_list[2]
        rel2 = main_route_list[3]
        ent3 = main_route_list[4]
        rel3 = main_route_list[5]
        ent4 = main_route_list[6]        
        triples1 = get_df(path, ent1, rel1, ent2)
        triples2 = get_df(path, ent2, rel2, ent3)
        triples3 = get_df(path, ent3, rel3, ent4)
        
        if ent1 in [ent2, ent3, ent4]:
            ent1 = ent1 + "_a"
            triples1.columns = [ent1, rel1, ent2]
        if ent2 in [ent2, ent3, ent4]:
            ent2 = ent2 + "_b"
            triples2.columns = [ent2, rel2, ent3]
            triples1.columns = [ent1, rel2, ent2]
        if ent3 in [ent1, ent2, ent4]:
            ent3 = ent3 + "_c"
            triples3.columns = [ent3, rel3, ent4]
            triples2.columns = [ent2, rel2, ent3]
        if ent4 in [ent1, ent2, ent3]:
            ent4 = ent4 + "_d"
            triples3.columns = [ent3, rel3, ent4]
            
        result = pd.merge(triples1, triples2, on = ent2)
        result = pd.merge(result, triples3, on = ent3)
    
    return result
        
        
def add_constraint(main_df, constraints):
    constraint_list = constraints.split("|")
    for constraint in constraint_list:
        constraint_parts = constraint.split("-")
        constraint_h = constraint_parts[0]
        constraint_r = constraint_parts[1]
        constraint_t = constraint_parts[2]
        constraint_df = get_df(triples_path, constraint_h, constraint_r, constraint_t)
        if len(constraint_list) == 4:
            place = constraint_list[-1]
            constraint_h = constraint_h + place
            constraint_df.columns = [constraint_h, constraint_r, constraint_t]
        for item in main_df.columns():
            if item == constraint:
                to_merge = item
        main_df = pd.merge(main_df, constraint_df, on = to_merge)
        
    return main_df

def get_qa_sample(main_df, main_route, constraints):
    main_entities = main_route.split("-")
    to_check = main_entities[1:]
    head = main_entities[0]
    if head in to_check:
        head = head + "_a"
    constraint_tails = []
    constraint_list = constraints.split("|")
    for item in constraint_list:
        constraint_tails.append(item.split("-")[2])
    sample_from = main_df[constraint_tails].drop_duplicates()

    if "date" in constraint_tails:
        temp = sample_from[[head, "date"]].drop_duplicates()
        counts = temp[head].value_counts()
        sample_from[sample_from[head].isin(counts.index[counts > 1])]
        
    
        
        
        
        
        
        
        
        


def write_df_qa(qa, file_path, qa_filename,  question):
    #print(qa)
    qa_df = pd.DataFrame.from_dict(qa, orient = "index")
    qa_df.reset_index(inplace = True)
    qa_df.columns = ["q", "a"]
    
    qa_df.to_csv(file_path + qa_filename, sep = "\t", index = False, header = None)
    #qa_df.to_csv("qa/1hop/csv/" + qa_filename + ".csv", index=False, header = None, encoding = "utf-8")
    #qa_df.to_csv("qa/1hop/txt/" + qa_filename + ".txt", sep = '\t', index = False, header = None)
    #qa_df.to_csv('qa/1hop/qa_1hop.txt', sep = '\t', mode = 'a', index = False, header = None)
    
    #record = main_route + "\t" + constraints + "\t" + question + "\t" + str(len(qa_df)) + "\n"

    #with open("qa_templates/qa_dataset_overview.txt", "a") as myfile:
        #myfile.write(record)
        
def write_qa_record(record_file, entity_route, question, index, qa_length, constraint):
    print("this is constraint:", constraint)
    record = str(index) + "\t" + entity_route + "\t" + constraint + "\t" + question + "\t" + str(qa_length) + "\n"
    print("this is record:", record)
    with open(record_file, "a") as myfile:
        myfile.write(record)


#********************************************************************************
#********************************fkr one hop questions***************************
#********************************************************************************

# path is template path
def pick_answer_1hop(temp_path, triples_path, head, rel, tail, question, sample_size, qa_filename, constraint = None):
    
    
    qa = dict()
    
    # main part of question
    main_df = get_df(triples_path, head, rel, tail)
    
    if tail == "price"  or tail == "count":
        main_df[tail] = main_df[tail].apply(lambda x: str(x))
    
         
    # take a sample
    sample_from = list(set(main_df[head].to_list()))
    random.shuffle(sample_from)
    
    if len(sample_from) >= sample_size:
        sample = random.sample(sample_from, sample_size)
    else:
        sample = sample_from

                
    for item in sample:
        a = "|".join(list(set(main_df.loc[main_df[head] == item, tail].to_list())))
        item = turn_into_readable(head, item)
        q = question.replace("[" + head + "]", item)
        qa.update({q:a})
        
    return qa
    
    
    
def pick_answer_1hop_constraint(temp_path, triples_path, head, rel, tail, question, sample_size, qa_filename, constraint):

    qa = dict()
    
    # main part of question
    main_df = get_df(triples_path, head, rel, tail)
    
    if tail == "price"  or tail == "count":
        main_df[tail] = main_df[tail].apply(lambda x: str(x))
    
     
    print("from utils:", head, rel, tail, constraint)
    constraint_parts = constraint.split("-")
    constraint_h = constraint_parts[0]
    constraint_r = constraint_parts[1]
    constraint_t = constraint_parts[2]
    
    # constraint part
    constraint_df = get_df(triples_path, constraint_h, constraint_r, constraint_t)
    
    
    
    if constraint_h == head:
        join_column = head
    elif constraint_h == tail:
        join_column = tail
    print(join_column)
    print(main_df)
    print(constraint_df)
    main_df = pd.merge(main_df, constraint_df, on = join_column)
    print(main_df)
        
    sample_from = main_df[[head, constraint_t]].drop_duplicates()
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from
        
        
    for index, row in sample.iterrows():
        mask = (main_df[head] == row[head]) & (main_df[constraint_t] == row[constraint_t])
        result_df = main_df[mask]
        a = "|".join(list(set(result_df[tail].to_list())))
        row_head = turn_into_readable(head, row[head])
        
        q = question.replace("[" + head + "]", row_head)
        row_constraint = row[constraint_t]
        q = q.replace("(" + constraint_t + ")", row_constraint)
        qa.update({q:a})
        
    return qa



def pick_answer_1hop_temporal(temp_path, triples_path, head, rel, tail, question, sample_size, qa_filename, constraint, year_only = False):
    
    months = {"01":"January", "02":"February", "03":"March", "04":"April", "05":"May", "06":"June", "07":"July", "08":"August", "09":"September", "10":"October", "11":"November", "12":"December"}
    qa = dict()
    
    # main part of question
    main_df = get_df(triples_path, head, rel, tail)
    
    if tail == "price"  or tail == "count":
        main_df[tail] = main_df[tail].apply(lambda x: str(x))
    
        
    print("from utils:", head, rel, tail, constraint)
    constraint_parts = constraint.split("-")
    constraint_h = constraint_parts[0]
    constraint_r = constraint_parts[1]
    constraint_t = constraint_parts[2]
    
    # constraint part
    constraint_df = get_df(triples_path, constraint_h, constraint_r, constraint_t)
    
    if constraint_h == head:
        join_column = head
    elif constraint_h == tail:
        join_column = tail
        
    print(join_column)
    print(main_df)
    print(constraint_df)
    main_df = pd.merge(main_df, constraint_df, on = join_column)
    print(main_df)
    
    if year_only == True:
        qa_filename = qa_filename + "year_only"
        main_df[constraint_t] = main_df[constraint_t].apply(lambda x: x[-4:])    
        
     
    sample_from = main_df[[head, constraint_t]].drop_duplicates()
    counts = sample_from[head].value_counts()
    sample_from = sample_from[sample_from[head].isin(counts.index[counts > 1])]
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from
        

    for index, row in sample.iterrows():
        mask = (main_df[head] == row[head]) & (main_df[constraint_t] == row[constraint_t])
        result_df = main_df[mask]
        a = "|".join(list(set(result_df[tail].to_list())))
        row_head = turn_into_readable(head, row[head])
        q = question.replace("[" + head + "]", row_head)
        q = q.replace("(" + constraint_t + ")", row[constraint_t])
        qa.update({q:a})
        
    return qa



#********************************************************************************
#*******************************2 hop questions**********************************
#********************************************************************************

def pick_answer_2hop(temp_path, triples_path, ent1, rel1, ent2, rel2, ent3, question, sample_size, qa_filename, constraint = None):
    print("this is question:", question)
    qa = dict()
    print(ent1, rel1, ent2, rel2, ent3)
    print(question)
    question_head = ent1
    
    triples1 = get_df(triples_path, ent1, rel1, ent2)
    triples2 = get_df(triples_path, ent2, rel2, ent3)
        
    print(triples1)
    print(triples2)

    if ent1 == ent3:
        triples1 = triples1.rename(columns = {ent1 : ent1 + "_a"})
        triples2 = triples2.rename(columns = {ent3 : ent3 + "_c"})
        ent1 = ent1 + "_a"
        ent3 = ent3 + "_c"

    main_df = pd.merge(triples1, triples2, on = ent2)
    print("merged")
    
    if ent3 == "price"  or ent3 == "count":
        main_df[ent3] = main_df[ent3].apply(lambda x: str(x))

    head = ent1
    tail = ent3

    print(head, tail)
    
        
    # take a sample
    sample_from = list(set(main_df[head].to_list()))
    
    print(head)
    print("sample taken")
    random.shuffle(sample_from)
    
    if len(sample_from) >= sample_size:
        sample = random.sample(sample_from, sample_size)
    else:
        sample = sample_from
        
    print("sample made:")
    print(sample)
                
    for item in sample:
        a = "|".join(list(set(main_df.loc[main_df[head] == item, tail].to_list())))
        item = turn_into_readable(head, item)
        q = question.replace("[" + question_head + "]", item)
        a = a.strip('"')
        qa.update({q:a})
        
    return qa
        
 
def pick_answer_2hop_constraint(triples_path, ent1, rel1, ent2, rel2, ent3, question, sample_size, constraint):

    print("this is question:", question)
    qa = dict()
    print(ent1, rel1, ent2, rel2, ent3)
    question_head = ent1
    
    triples1 = get_df(triples_path, ent1, rel1, ent2)
    triples2 = get_df(triples_path, ent2, rel2, ent3)
    

    if ent1 == ent3:
        triples1 = triples1.rename(columns = {ent1 : ent1 + "_a"})
        triples2 = triples2.rename(columns = {ent3 : ent3 + "_c"})
        ent1 = ent1 + "_a"
        ent3 = ent3 + "_c"

    main_df = pd.merge(triples1, triples2, on = ent2)
    
    if ent3 == "price"  or ent3 == "count":
        main_df[ent3] = main_df[ent3].apply(lambda x: str(x))

    head = ent1
    tail = ent3
    
    print("this is constraint:", constraint)
    if "(" in constraint:
        constraint_loc = constraint.split("(")[1].strip(")")
        constraint = constraint[:-3]
        constraint_parts = constraint.split("-")
        constraint_h = constraint_parts[0]
        constraint_r = constraint_parts[1]
        constraint_t = constraint_parts[2]
        
        # constraint part
        constraint_df = get_df(triples_path, constraint_h, constraint_r, constraint_t)
        constraint_df = constraint_df.rename(columns = {constraint_h : constraint_h + "_" + constraint_loc})
        constraint_h = constraint_h + "_" + constraint_loc
    else:
        constraint_parts = constraint.split("-")
        constraint_h = constraint_parts[0]
        constraint_r = constraint_parts[1]
        constraint_t = constraint_parts[2]
        
        # constraint part
        constraint_df = get_df(triples_path, constraint_h, constraint_r, constraint_t)

    if constraint_h == ent1:
        join_column = ent1
    elif constraint_h == ent2:
        join_column = ent2
    elif constraint_h == ent3:
        join_column = ent3
        
    main_df = pd.merge(main_df, constraint_df, on = join_column)
        
    sample_from = main_df[[head, constraint_t]].drop_duplicates()
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from
        
    for index, row in sample.iterrows():
        mask = (main_df[head] == row[head]) & (main_df[constraint_t] == row[constraint_t])
        result_df = main_df[mask]
        a = "|".join(list(set(result_df[tail].to_list())))
        row_head = turn_into_readable(question_head, row[head])
        q = question.replace("[" + question_head + "]", row_head)
        row_constraint = row[constraint_t]
        row_constraint = turn_into_readable(constraint_t, row[constraint_t])
        q = q.replace("(" + constraint_t + ")", row_constraint)
        qa.update({q:a})
        
    return qa


#********************************************************************************
#*******************************advanced questions**********************************
#********************************************************************************


def get_q2_1(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "org", "launched", "funding_round")
    triples2 = get_df("triples/", "org", "funding_round_investor", "funding_round")
    triples3 = get_df("triples/", "person", "funding_round_investor", "funding_round")
    triples2 = triples2.rename(columns = {"org":"investor"})
    triples3 = triples3.rename(columns = {"person":"investor"})
    triples4 = triples2.append(triples3)
    main_df = pd.merge(triples1, triples4, on = "funding_round")
    # take a sample
    sample_from = main_df[["org", "investor"]].drop_duplicates()["org"].value_counts().rename_axis('org').reset_index(name='counts')
    sample_from = sample_from[sample_from["counts"] > 2]
    sample_from = sample_from["org"].to_list()
    #random.shuffle(sample_from)
    
    if len(sample_from) >= sample_size:
        sample = random.sample(sample_from, sample_size)
    else:
        sample = sample_from
                
    for item in sample:
        a = "|".join(list(set(main_df.loc[main_df["org"] == item, "investor"].to_list())))
        item = turn_into_readable("org", item)
        q = question.replace("[" + "org" + "]", item)
        qa.update({q:a})
        
    return qa

def get_q2_2_1(question, sample_size, qa_filename):
    qa = dict()
    p = ["sponsor", "speaker", "exhibitor", "organizer", "contestant"]
    main_df = pd.DataFrame()
    for item in p:
        temp = get_df("triples/", "org", item, "event")
        main_df = main_df.append(temp)
    main_df = main_df.drop_duplicates()
    triples2 = get_df("triples/", "event", "type_of", "event_role")
    main_df = pd.merge(main_df, triples2, on = "event")
    main_df = main_df[["org", "event", "event_role"]].drop_duplicates()
    
    
    print(main_df)
    
    sample_from = main_df["org"].value_counts().rename_axis('org').reset_index(name='counts')
    sample_from = sample_from[sample_from["counts"] >= 2]
    sample_from = sample_from["org"].to_list()
    #random.shuffle(sample_from)
    
    if len(sample_from) >= sample_size:
        sample = random.sample(sample_from, sample_size)
    else:
        sample = sample_from
                
    for item in sample:
        a = "|".join(list(set(main_df.loc[main_df["org"] == item, "event_role"].to_list())))
        item = turn_into_readable("org", item)
        q = question.replace("[" + "org" + "]", item)
        qa.update({q:a})
        
    return qa
    
    
def get_q2_2(question, sample_size, qa_filename):
    qa = dict()
    p = ["sponsor", "speaker", "exhibitor", "organizer", "contestant"]
    main_df = pd.DataFrame()
    for item in p:
        temp = get_df("triples/", "org", item, "event")
        main_df = main_df.append(temp)
    main_df = main_df.drop_duplicates()
    triples2 = get_df("triples/", "event", "type_of", "event_role")
    main_df = pd.merge(main_df, triples2, on = "event")
    main_df = main_df[["org", "event", "event_role"]].drop_duplicates()
    
    print(main_df)
    
    sample_from = main_df["org"].value_counts().rename_axis('org').reset_index(name='counts')
    sample_from = sample_from[sample_from["counts"] >= 2]
    sample_from = sample_from["org"].to_list()
    #random.shuffle(sample_from)
    
    if len(sample_from) >= sample_size:
        sample = random.sample(sample_from, sample_size)
    else:
        sample = sample_from
                
    for item in sample:
        a = "|".join(list(main_df.loc[main_df["org"] == item, "event_role"].mode()))
        item = turn_into_readable("org", item)
        q = question.replace("[" + "org" + "]", item)
        qa.update({q:a})
        
    return qa


def get_q2_3_1(question, sample_size, qa_filename):
    qa = dict()
    p = ["sponsor", "speaker", "exhibitor", "organizer", "contestant"]
    main_df = pd.DataFrame()
    for item in p:
        temp = get_df("triples/", "person", item, "event")
        main_df = main_df.append(temp)
    main_df = main_df.drop_duplicates()
    triples2 = get_df("triples/", "event", "type_of", "event_role")
    main_df = pd.merge(main_df, triples2, on = "event")
    main_df = main_df[["person", "event", "event_role"]].drop_duplicates()
    
    
    print(main_df)
    
    sample_from = main_df["person"].value_counts().rename_axis('person').reset_index(name='counts')
    sample_from = sample_from[sample_from["counts"] >= 2]
    sample_from = sample_from["person"].to_list()
    #random.shuffle(sample_from)
    
    if len(sample_from) >= sample_size:
        sample = random.sample(sample_from, sample_size)
    else:
        sample = sample_from
                
    for item in sample:
        a = "|".join(list(set(main_df.loc[main_df["person"] == item, "event_role"].to_list())))
        item = turn_into_readable("person", item)
        q = question.replace("[" + "person" + "]", item)
        qa.update({q:a})
        
    return qa
    
def get_q2_3(question, sample_size, qa_filename):
    qa = dict()
    p = ["sponsor", "speaker", "exhibitor", "organizer", "contestant"]
    main_df = pd.DataFrame()
    for item in p:
        temp = get_df("triples/", "person", item, "event")
        main_df = main_df.append(temp)
    main_df = main_df.drop_duplicates()
    triples2 = get_df("triples/", "event", "type_of", "event_role")
    main_df = pd.merge(main_df, triples2, on = "event")
    main_df = main_df[["person", "event", "event_role"]].drop_duplicates()
    
    print(main_df)
    
    sample_from = main_df["person"].value_counts().rename_axis('person').reset_index(name='counts')
    sample_from = sample_from[sample_from["counts"] >= 2]
    sample_from = sample_from["person"].to_list()
    #random.shuffle(sample_from)
    
    if len(sample_from) >= sample_size:
        sample = random.sample(sample_from, sample_size)
    else:
        sample = sample_from
                
    for item in sample:
        a = "|".join(list(main_df.loc[main_df["person"] == item, "event_role"].mode()))
        item = turn_into_readable("person", item)
        q = question.replace("[" + "person" + "]", item)
        qa.update({q:a})
        
    return qa

def get_q2_4(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "org", "category", "category")
    triples2 = get_df("triples/", "org", "org_ipo", "ipo")
    triples3 = get_df("triples/", "ipo", "went_public_on", "date")
    main_df = pd.merge(triples1, triples2, on = "org")
    main_df = pd.merge(main_df, triples3, on = "ipo")
    main_df["date"] = main_df["date"].apply(lambda x: x[-4:])
    main_df = main_df[["category", "org", "date"]].drop_duplicates()
    print(main_df)
    sample_from = main_df[["category", "date"]].drop_duplicates()
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from
                
    for index, row in sample.iterrows():
        
        mask = (main_df["category"] == row["category"]) & (main_df["date"] == row["date"])
        a = "|".join(list(set(main_df[mask]["org"].to_list())))
        item = turn_into_readable("category", row["category"])
        q = question.replace("[" + "category" + "]", item)
        q = q.replace("(date)", row["date"])
        qa.update({q:a})
        
    return qa
    
def get_q2_5(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "job", "in_org", "org")
    triples2 = get_df("triples/", "person", "has_job", "job")
    triples3 = get_df("triples/", "job", "is_current", "job_current")
    triples3 = triples3[triples3["job_current"] == False]
    main_df = pd.merge(triples1, triples2, on = "job")
    main_df = pd.merge(main_df, triples3, on = "job")
    main_df = main_df.rename(columns = {"job": "job_a"})
    #print(main_df)    
    triples4 = get_df("triples/", "job", "has_job", "person")
    triples5 = get_df("triples/", "job", "job_title", "job_title")
    triples5 = triples5[triples5["job_title"] == "founder"]
    triples6 = get_df("triples/", "job", "in_org", "org")
    triples6 = triples6.rename(columns = {"org":"org_target"})
    constraint = pd.merge(triples4, triples5, on = "job")
    constraint = pd.merge(constraint, triples6, on = "job")
    main_df = pd.merge(main_df, constraint, on = "person")[["org", "person", "org_target"]].drop_duplicates()
    #print(main_df)
    
    found_count = main_df[["person", "org_target"]].drop_duplicates()
    sample_from = found_count["person"].value_counts().rename_axis('person').reset_index(name='counts')
    sample_from = sample_from[sample_from["counts"] >= 2]
    #random.shuffle(sample_from)
    # make sure each person founded more than 1 companies
    sample_from = pd.merge(main_df, sample_from, on = "person")
    print(sample_from)
    print(sample_from.columns)
    print(sample_from.counts.unique())
    sample = list(set(sample_from.org.to_list()))
    
    if len(sample_from) >= sample_size:
        sample = random.sample(sample, sample_size)               
    
    for item in sample:
        temp = sample_from[sample_from["org"] == item]
        num = random.sample(list(temp.counts.unique()), 1)[0]
        temp_people = temp[temp["counts"] >= num].person.to_list()
        a = "|".join(list(set(temp_people)))
        item = turn_into_readable("org", item)
        q = question.replace("[" + "org" + "]", item)
        q = q.replace("(num)", str(num))
        qa.update({q:a})
        
    return qa
    
def get_q2_6(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "org", "is_acquirer", "acquisition")
    triples2 = get_df("triples/", "acquisition", "type_of", "acquisition_type")
    main_df = pd.merge(triples1, triples2, on = "acquisition").drop_duplicates()
    main_df['counts'] = main_df.groupby(['org','acquisition_type'])['acquisition_type'].transform('count')
    main_df = main_df[main_df["counts"] > 1]
    sample_from = main_df[["org", "acquisition_type"]].drop_duplicates()
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from
        
    print(main_df) 
    for index, row in sample.iterrows():
        mask = (main_df["org"] == row["org"]) & (main_df["acquisition_type"] == row["acquisition_type"])
        a = str(len(main_df[mask].acquisition.unique()))
        item1 = turn_into_readable("org", row["org"])
        item2 = turn_into_readable("acquisition_type", row["acquisition_type"])
        q = question.replace("[org]", item1)
        q = q.replace("(acquisition_type)", item2)
        
        qa.update({q:a})
        
    return qa

def get_q2_7(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "category", "category", "org")
    triples2 = get_df("triples/", "org", "founded_on", "date")
    triples2["date"] = triples2["date"].apply(lambda x: x[-4:])
    triples2["date"] = triples2["date"].apply(lambda x: int(x))
    main_df = pd.merge(triples1, triples2, on = "org").drop_duplicates()
    
    sample_from = main_df[["category", "org", "date"]].drop_duplicates()
    sample_from['counts'] = sample_from.groupby(['category','date'])['category'].transform('count')
    # ensure there is a time range
    print(sample_from)
    sample_from = sample_from[sample_from["counts"] > 2]
    sample = sample_from.sample(frac = 1)
        
    for index, row in sample.iterrows():
        temp = main_df[main_df["category"] == row["category"]]
        years = random.sample(range(2010, 2021), 2)
        years = sorted(years)
        y1 = years[1]
        y2 = random.sample(range(y1, 2022), 1)[0]
        print(y1, y2)
            

        mask = (temp["date"] >= y1) & (temp["date"] <= y2)
        a = "|".join(list(set(temp[mask].org.to_list())))
        category = turn_into_readable("category", row["category"])        
        q = question.replace("[category]", category)
        q = q.replace("(year1)", str(y1))
        q = q.replace("(year2)", str(y2))
        qa.update({q:a})
        if len(qa) >= sample_size:
            break
    return qa   
    
def get_q3_1(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "org", "is_acquirer", "acquisition")
    triples1 = triples1.rename(columns = {"org":"org_acquirer"})
    triples2 = get_df("triples/", "org", "is_acquiree", "acquisition")
    triples3 = get_df("triples/", "org", "category", "category")
    main_df = pd.merge(triples1, triples2, on = "acquisition")
    main_df = pd.merge(main_df, triples3, on = "org").drop_duplicates()
    main_df = main_df[["org_acquirer", "org", "category"]].drop_duplicates()
    sample_from = main_df[["org_acquirer", "category"]].drop_duplicates()
    # ensure the sample has 2 or more category
    sample_from = sample_from["org_acquirer"].value_counts().rename_axis('org_acquirer').reset_index(name='counts')
    sample_from = list(set(sample_from[sample_from["counts"] >= 2]["org_acquirer"].to_list()))
    
    
    if len(sample_from) >= sample_size:
        sample = random.sample(sample_from, sample_size)
    else:
        sample = sample_from
        
    for item in sample:
        a = "|".join(list(main_df.loc[main_df["org_acquirer"] == item, "category"].mode()))
        #print(a)
        #print(main_df.loc[main_df["org_acquirer"] == item, "category"].to_list())
        item = turn_into_readable("org", item)
        q = question.replace("[org]", item)
        qa.update({q:a})
    return qa

def get_q3_2(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "event", "city", "city")
    triples2 = get_df("triples/", "event", "started_on", "date")
    triples2["date"] = triples2["date"].apply(lambda x: x[-4:])
    triples3 = get_df("triples/", "event", "type_of", "event_role")
    main_df = pd.merge(triples1, triples2, on = "event")
    main_df = pd.merge(main_df, triples3, on = "event").drop_duplicates()
    sample_from = main_df[["event_role", "city", "date"]].drop_duplicates()
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from.sample(n = len(sample))
        
    for index, row in sample.iterrows():
        date = row["date"]
        city = row["city"]
        event_role = row["event_role"]
        mask = (main_df["event_role"] == event_role) & (main_df["city"] == city) & (main_df["date"] == date)
        
        a = "|".join(list(set(main_df[mask]["event"].to_list())))
        q = question.replace("[event_role]", event_role)
        q = q.replace("(date)", date)
        q = q.replace("(city)", city)
        
        qa.update({q:a})
        
    return qa
    
def get_num(x):
    if x == "10000+":
        x = int(x.strip("+"))
    
    else:
        x = int(x.split("-")[0]) - 1
    return x
        
        
def get_q3_3(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "org", "category", "category")
    triples2 = get_df("triples/", "org", "employee_count", "employee_count")
    triples2["employee_count"] = triples2["employee_count"].apply(lambda x: get_num(x))
    triples3 = get_df("triples/", "org", "country_code", "country_code")
    main_df = pd.merge(triples1, triples2, on = "org")
    main_df = pd.merge(main_df, triples3, on = "org").drop_duplicates()
    sample_from = main_df[["category", "country_code"]].drop_duplicates()
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from.sample(n = len(sample))
        
    for index, row in sample.iterrows():
        c = row["category"]
        coun = row["country_code"]
        mask = (main_df["category"] == c) & (main_df["country_code"] == coun)
        temp = main_df[mask]
        m = temp["employee_count"].max()
        a = "|".join(list(set(temp[temp["employee_count"] == m].org.to_list())))
        print(a)
        c = turn_into_readable("category", c)
        q = question.replace("[category]", c)
        coun = turn_into_readable("country_code", coun)
        q = q.replace("(country_code)", coun)
        qa.update({q:a})
    return qa
    
def get_q3_4(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "org", "is_acquiree", "acquisition")
    triples2 = get_df("triples/", "org", "category", "category")
    triples3 = get_df("triples/", "acquisition", "acquired_on", "date")
    triples3["date"] = triples3["date"].apply(lambda x: int(x[-4:]))
    main_df = pd.merge(triples1, triples2, on = "org")
    main_df = pd.merge(main_df, triples3, on = "acquisition").drop_duplicates()
    
    sample_from = main_df[["category", "date"]].drop_duplicates()
    # recent
    sample_from = sample_from[sample_from["date"] >= 2010]
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from.sample(n = len(sample))
        
    for index, row in sample.iterrows():
        c = row["category"]
        d = row["date"]
        mask = (main_df["category"] == c) & (main_df["date"] == d)
        temp = main_df[mask]
        a = "|".join(list(set(temp.org.to_list())))
        c = turn_into_readable("category", c)
        q = question.replace("[category]", c)
        q = q.replace("(date)", str(d))
        qa.update({q:a})
    return qa
    
def get_q3_5(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "org", "launched", "funding_round")
    triples2 = get_df("triples/", "funding_round", "type_of", "investment_type")
    triples3 = get_df("triples/", "org", "funding_round_investor", "funding_round")
    triples3 = triples3.rename(columns = {"org":"investor"})
    triples4 = get_df("triples/", "person", "funding_round_investor", "funding_round")
    triples4 = triples4.rename(columns = {"person" : "investor"})
    triples5 = triples3.append(triples4)
    main_df = pd.merge(triples1, triples2, on = "funding_round")
    main_df = pd.merge(main_df, triples5, on = "funding_round").drop_duplicates()
    sample_from = main_df[["org", "investment_type"]].drop_duplicates()
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from.sample(n = len(sample))
        
    for index, row in sample.iterrows():
        o = row["org"]
        t = row["investment_type"]
        mask = (main_df["org"] == o) & (main_df["investment_type"] == t)
        temp = main_df[mask]
        a ="|".join(list(set(temp.investor.to_list())))
        o = turn_into_readable("org", o)
        t = turn_into_readable("investment_type", t)
        q = question.replace("[org]", o)
        q = q.replace("(investment_type)", t)
        
        qa.update({q:a})
    return qa
        
def get_q3_6(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "person", "gender", "gender")
    triples2 = get_df("triples/", "person", "has_job", "job")
    triples3 = get_df("triples/", "job","job_title", "job_title")
    triples4 = get_df("triples/", "job", "in_org", "org")
    main_df = pd.merge(triples1, triples2, on = "person")
    main_df = pd.merge(main_df, triples3, on = "job")
    main_df = main_df[main_df["job_title"] == "founder"]
    main_df = pd.merge(main_df, triples4, on = "job").drop_duplicates()
    p = ["sponsor", "speaker", "exhibitor", "organizer", "contestant"]
    triples5 = pd.DataFrame()
    for item in p:
        temp = get_df("triples/", "org", item, "event")
        triples5 = triples5.append(temp)
    main_df = pd.merge(main_df, triples5, on = "org").drop_duplicates()
    main_df = main_df[["gender", "org", "event"]]
    print(main_df)
    all_gender = list(main_df.gender.unique())
    print(all_gender)
    #print(main_df[main_df["gender"] == "non_binary"])
    
    gender_proportion = 0.7
    
    minority_gender = []
    for gender in all_gender:
        if (gender != "female") & (gender != "male"):
            minority_gender.append(gender)
            
            
    for gender in minority_gender:
        temp = main_df[main_df["gender"] == gender]
        print(temp.columns)
        event = random.sample(list(set(temp.event.to_list())), 1)[0]
        a = "|".join(temp[temp["event"] == event]["org"])
        event = turn_into_readable("event", event)
        q = question.replace("[event]", event)
        gender = turn_into_readable("gender", gender)
        q = q.replace("(gender)", gender)
        qa.update({q:a})
        
    female_sample_size = int(sample_size*gender_proportion) - len(qa)
    female_main_df = main_df[main_df["gender"] == "female"]
    female_sample = random.sample(list(set(female_main_df.event.to_list())), female_sample_size)
    for item in female_sample:
        event = random.sample(list(set(female_main_df.event.to_list())), 1)[0]
        a = "|".join(female_main_df[female_main_df["event"] == event]["org"])
        event = turn_into_readable("event", event)
        q = question.replace("[event]", event)
        q = q.replace("(gender)", "female")
        qa.update({q:a})
        
    male_sample_size = sample_size - len(qa)
    male_main_df = main_df[main_df["gender"] == "male"]
    male_sample = random.sample(list(set(male_main_df.event.to_list())), male_sample_size)
    for item in male_sample:
        event = random.sample(list(set(male_main_df.event.to_list())), 1)[0]
        a = "|".join(female_main_df[female_main_df["event"] == event]["org"])
        event = turn_into_readable("event", event)
        q = question.replace("[event]", event)
        q = q.replace("(gender)", "male")
        qa.update({q:a})
        
    return qa
        
            
            
            
            
    
    
def get_q3_7(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "category", "category", "org")
    triples2 = get_df("triples/", "org", "org_ipo", "ipo")
    triples3 = get_df("triples/", "ipo", "share_price_usd", "price")
    main_df = pd.merge(triples1, triples2, on = "org")
    main_df = pd.merge(main_df, triples3, on = "ipo").drop_duplicates()
    print(main_df)
    
    sample_from = main_df[["category", "price"]].drop_duplicates()
    
    sample_from = sample_from["category"].value_counts().rename_axis('category').reset_index(name='counts')
    sample_from = list(set(sample_from[sample_from["counts"] >= 2]["category"].to_list()))
        
    if len(sample_from) >= sample_size:
        sample = random.sample(sample_from, sample_size)
    else:
        sample = sample_from
        
    for item in sample:
        temp = main_df[main_df["category"] == item][["org", "price"]].drop_duplicates()
        result = temp.loc[temp['price'] == temp['price'].max(), 'org'].to_list()
        #print(result)
        #print("*"*50)
        a = "|".join(result)
        item = turn_into_readable("category", item)
        q = question.replace("[category]", item)
        qa.update({q:a})
        
    return qa
    
def get_q3_8(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "org", "launched", "funding_round")
    triples2 = get_df("triples/", "funding_round", "type_of", "investment_type")
    triples3 = get_df("triples/", "funding_round", "funding_round_investor", "person")
    triples3 = triples3.rename(columns = {"person":"investor"})
    triples4 = get_df("triples/", "funding_round", "funding_round_investor", "org")
    triples4 = triples4.rename(columns = {"org":"investor"})
    inv = triples3.append(triples4)
    main_df = pd.merge(triples1, triples2, on = "funding_round")
    main_df = pd.merge(main_df, inv, on = "funding_round")
    main_df = main_df[["org", "investment_type", "investor"]].drop_duplicates()
    sample_from = main_df[["org", "investment_type"]].drop_duplicates()
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from
        
    for index, row in sample.iterrows():
        mask = (main_df["org"] == row["org"]) & (main_df["investment_type"] == row["investment_type"])
        temp = main_df[mask]
        count = len(temp.investor.unique())
        a = str(count)
        org = turn_into_readable("org", row["org"])
        q = question.replace("[org]", org)
        inv = turn_into_readable("investment_type", row["investment_type"])
        q = q.replace("(investment_type)", inv)
        qa.update({q:a})
    return qa


def get_q4_1(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "org", "country_code", "country_code")
    triples2 = get_df("triples/", "org", "launched", "funding_round")
    triples3 = get_df("triples/", "org", "funding_round_investor", "funding_round")
    triples3 = triples3.rename(columns = {"org":"investor"})
    triples4 = get_df("triples/", "person", "funding_round_investor", "funding_round")
    triples4 = triples3.rename(columns = {"person":"investor"})
    triples5 = get_df("triples/", "org", "category", "category")
    inv = triples3.append(triples4)
    main_df = pd.merge(triples1, triples2, on = "org")
    main_df = pd.merge(main_df, triples5, on = "org")
    main_df = pd.merge(main_df, inv, on = "funding_round").drop_duplicates()
    sample_from = main_df[["category", "country_code"]]
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from
        
    for index, row in sample.iterrows():
        c = row["category"]
        coun = row["country_code"]
        mask = (main_df["country_code"] == coun) & (main_df["category"] == c)
        temp = main_df[mask]
        a = "|".join(list(set(temp.investor.to_list())))
        c = turn_into_readable("category", c)
        q = question.replace("[category]", c)
        coun = turn_into_readable("country_code", coun)
        q = q.replace("(country_code)", coun)
        qa.update({q:a})
    return qa
    
def get_q4_2(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "person", "gender", "gender")
    triples2 = get_df("triples/", "person", "has_job", "job")
    triples3 = get_df("triples/", "job", "job_title", "job_title")
    triples4 = get_df("triples/", "job", "in_org", "org")
    triples5 = get_df("triples/", "org", "employee_count", "employee_count")
    triples5["employee_count"] = triples5["employee_count"].apply(lambda x: get_num(x))
    
    triples6 = get_df("triples/", "job", "is_current", "job_current")
    main_df = pd.merge(triples1, triples2, on = "person")
    main_df = pd.merge(main_df, triples6, on = "job")
    print(main_df.columns)
    main_df = main_df[main_df["job_current"] == True]
    main_df = pd.merge(main_df, triples3, on = "job")
    main_df = pd.merge(main_df, triples4, on = "job")
    main_df = pd.merge(main_df, triples5, on = "org").drop_duplicates()
    
    all_gender = list(main_df.gender.unique())
    print(all_gender)
    
    #gender_proportion = 0.7
    
    minority_gender = []
    for gender in all_gender:
        if (gender != "female") & (gender != "male"):
            minority_gender.append(gender)
            
    for gender in minority_gender:
        temp = main_df[main_df["gender"] == gender]
        sample_from = temp[["job_title", "employee_count"]].drop_duplicates()
        sample = sample_from
        for index, row in sample.iterrows():
            j = row["job_title"]
            e = row["employee_count"]
            mask = (temp["job_title"] == j) & (temp["employee_count"] == e)
            a = "|".join(temp[mask]["person"])
            j = turn_into_readable("job_title", j)
            q = question.replace("[job_title]", j)
            gender = turn_into_readable("gender", gender)
            q = q.replace("(gender)", gender)
            q = q.replace("(employee_count)", str(e))
            qa.update({q:a})
        
        
    main_df = main_df[main_df["employee_count"] >= 100]
    sample_from = main_df[["person", "gender", "job_title", "employee_count"]].drop_duplicates()
    sample_from = sample_from[sample_from["employee_count"] > 1]
    sample = sample_from[["gender", "job_title", "employee_count"]].drop_duplicates()
    print(sample)
    for index, row in sample.iterrows():
        g = row["gender"]
        j = row["job_title"]
        e = row["employee_count"]
        mask = (main_df["gender"] == g) & (main_df["job_title"] == j) & (main_df["employee_count"] == e)
        a = "|".join(main_df[mask]["person"])
        q = question.replace("[job_title]", j)
        g = turn_into_readable("gender", g)
        q = q.replace("(gender)", g)
        q = q.replace("(employee_count)", str(e))
        qa.update({q:a})
   
    return qa
    
    
    
    
    
    
    
    

def get_q4_3(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "org", "category", "category")
    triples2 = get_df("triples/", "org", "launched", "funding_round")
    triples3 = get_df("triples/", "funding_round", "type_of", "investment_type")
    triples4 = get_df("triples/", "funding_round", "announced_on", "date")
    triples4["date"] = triples4["date"].apply(lambda x: x[-4:])
    main_df = pd.merge(triples1, triples2, on = "org")
    main_df = pd.merge(main_df, triples3, on = "funding_round")
    main_df = pd.merge(main_df, triples4, on = "funding_round").drop_duplicates()
    sample_from = main_df[["category", "investment_type", "date"]]

    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from
        
    for index, row in sample.iterrows():
        c = row["category"]
        i = row["investment_type"]
        d = row["date"]
        mask = (main_df["category"] == c) & (main_df["investment_type"] == i) & (main_df["date"] == d)
        
        temp = main_df[mask]
        a= "|".join(list(set(temp.org.to_list())))
        c = turn_into_readable("category", c)
        i = turn_into_readable("investment_type", i)

        q = question.replace("[category]", c)        
        q = q.replace("(investment_type)", i)
        q = q.replace("(date)", d)
        
        qa.update({q:a})
    return qa

        
def get_q4_4(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "person", "funding_round_investor", "funding_round")
    triples2 = get_df("triples/", "org", "funding_round_investor", "funding_round")
    triples3 = get_df("triples/", "person", "country_code", "country_code")
    triples4 = get_df("triples/", "org", "country_code", "country_code")
    temp1 = pd.merge(triples1, triples3, on = "person")
    temp1 = temp1.rename(columns = {"person":"investor"})
    temp2 = pd.merge(triples2, triples4, on = "org")
    temp2 = temp1.rename(columns = {"org":"investor"})
    temp3 = temp1.append(temp2)
    triples5 = get_df("triples/", "org", "launched", "funding_round")
    triples6 = get_df("triples/", "org", "category", "category")
    main_df = pd.merge(triples5, temp3, on = "funding_round")
    main_df = pd.merge(main_df, triples6, on = "org").drop_duplicates()
    sample_from = main_df[["country_code", "category"]].drop_duplicates()
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from
        
    for index, row in sample.iterrows():
        c = row["category"]
        coun = row["country_code"]
        mask = (main_df["country_code"] == coun) & (main_df["category"] == c)
        temp = main_df[mask]
        a = "|".join(list(set(temp.investor.to_list())))
        coun = turn_into_readable("country_code", coun)
        q = question.replace("[country_code]", coun)
        c = turn_into_readable("category", c)
        q = q.replace("(category)", c)
        qa.update({q:a})
    return qa
    
def get_q2_8(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "org", "is_acquirer", "acquisition")
    triples1 = triples1.rename(columns = {"org":"acquirer"})
    triples2 = get_df("triples/", "acquisition", "is_acquiree", "org")
    triples2 = triples2.rename(columns = {"org":"acquiree"})
    main_df = pd.merge(triples1, triples2, on = "acquisition")
    main_df = main_df[["acquirer", "acquiree"]].drop_duplicates()
    sample_from = list(set(main_df.acquirer.to_list()))
    
    if len(sample_from) >= sample_size:
        sample = random.sample(sample_from, sample_size)
    else:
        sample = sample_from
        
    for item in sample:
        temp = main_df[main_df["acquirer"] == item]
        a = str(len(temp.acquiree.unique()))
        item = turn_into_readable("org", item)
        q = question.replace("[org]", item)
        qa.update({q:a})
    return qa
        
def get_q3_10(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "category", "category", "org")
    triples2 = get_df("triples/", "org", "org_ipo", "ipo")
    triples3 = get_df("triples/", "ipo", "stock_exchange_symbol", "stock_exchange_symbol")
    main_df = pd.merge(triples1, triples2, on = "org")
    main_df = pd.merge(main_df, triples3, on = "ipo").drop_duplicates()
    sample_from = main_df[["category", "stock_exchange_symbol"]].drop_duplicates()
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from
        
    for index, row in sample.iterrows():
        c = row["category"]
        s = row["stock_exchange_symbol"]
        mask = (main_df["category"] == c) & (main_df["stock_exchange_symbol"] == s)
        temp = main_df[mask]
        a = "|".join(list(set(temp.org.to_list())))
        c = turn_into_readable("category", c)
        q = question.replace("[category]", c)
        q = q.replace("(stock_exchange_symbol)", s)
        qa.update({q:a})
    return qa

def get_q4_5(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "org", "launched", "funding_round")
    triples2 = get_df("triples/", "org", "country_code", "country_code")
    temp1 = pd.merge(triples1, triples2, on = "org")
    temp1 = temp1.rename(columns = {"country_code":"country_code_org"})
    triples3 = get_df("triples/", "org", "lead_investor_of", "funding_round")
    triples4 = get_df("triples/", "person", "lead_investor_of", "funding_round")
    triples5 = get_df("triples/", "org", "country_code", "country_code")
    triples6 = get_df("triples/", "person", "country_code", "country_code")
    temp2 = pd.merge(triples3, triples5, on = "org")
    temp3 = pd.merge(triples4, triples6, on = "person")
    temp2 = temp2.rename(columns = {"org":"investor"})
    temp3 = temp3.rename(columns = {"person": "investor"})
    temp4 = temp2.append(temp3)
    temp4 = temp4.rename(columns = {"country_code":"country_code_inv"})
    main_df = pd.merge(temp1, temp4, on = "funding_round").drop_duplicates()
    print(main_df)
    print(main_df.columns)
    
    sample_from = main_df[["country_code_org", "country_code_inv"]].drop_duplicates()
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from
        
    for index, row in sample.iterrows():
        c_org = row["country_code_org"]
        c_inv = row["country_code_inv"]
        mask = (main_df["country_code_org"] == c_org) & (main_df["country_code_inv"] == c_inv)
        temp = main_df[mask]
        a = "|".join(list(set(temp.investor.to_list())))
        c_org = turn_into_readable("country_code", c_org)
        c_inv = turn_into_readable("country_code", c_inv)
        q = question.replace("[country_code]", c_org)
        q = q.replace("(country_code)", c_inv)
        qa.update({q:a})
    return qa
    
    
def get_q3_9(question, sample_size, qa_filename):
    qa = dict()
    triples1 = get_df("triples/", "category", "category", "org")
    triples2 = get_df("triples/", "funding_round", "launched", "org")
    triples3 = get_df("triples/", "funding_round", "raised_amount_usd", "price")
    main_df = pd.merge(triples1, triples2, on = "org")
    main_df = pd.merge(main_df, triples3, on = "funding_round").drop_duplicates()
   
    main_df['org_sum'] = main_df.groupby('org')['price'].transform('sum')
    main_df['maxi'] = main_df.groupby('category')['org_sum'].transform('max')
    
    amounts = [100000000, 500000000, 1000000000, 5000000000, 10000000000, 50000000000, 100000000000, float("inf")]
    
    main_df = main_df[main_df["maxi"] > amounts[0]]
    

    sample_from = main_df[["category", "org_sum"]].drop_duplicates()
    sample_from = sample_from[sample_from["org_sum"] > amounts[0]]
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from
        
    
    for index, row in sample.iterrows():
        c = row["category"]
        choice = row["org_sum"]
        for i in range(len(amounts)-1):
            if choice >= amounts[i] and choice < amounts[i+1]:
                thresh = amounts[i]
        try:
            print(thresh)
        except:
            print("choice")
            print(choice)
        mask = (main_df["category"] == c) & (main_df["org_sum"] >= thresh)
        answer_df = main_df[mask]
        a = "|".join(list(set(answer_df.org.to_list())))
        c = turn_into_readable("category", c)
        q = question.replace("[category]", c)
        q = q.replace("(price)", str(thresh))
        qa.update({q:a})
    return qa
    
        
        
        
    
    
    
    
    
    
    
    
    
    
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
    
        
        
    
'''
q = question
testing = pd.read_csv("triples/funding_round-announced_on-date.csv")

testing["date"] = testing["date"].apply(lambda x: x[-4:])
testing["type"] = testing["date"].apply(lambda x: type(x))

print(testing)
'''