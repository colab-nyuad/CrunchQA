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
cluster_file = open("clustering/clusters.pickle", "rb")
cluster_centers = pickle.load(cluster_file)
cluster_file.close()
#print(cluster_centers)

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
    elif (entity == "event_role") or (entity == "category") or (entity == "category_group"):
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
    
    if (tail == "count") or (tail == "price"):
        main_df[tail] = main_df[tail].apply(lambda x: str(cluster_centers[x][2]))
        
       
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
    
    if (tail == "count") or (tail == "price"):
        main_df[tail] = main_df[tail].apply(lambda x: str(cluster_centers[x][2]))
    
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
    
    if (tail == "count") or (tail == "price"):
        main_df[tail] = main_df[tail].apply(lambda x: str(cluster_centers[x][2]))
        
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
    counts = sample_from[head].value_counts()
    sample_from = sample_from[sample_from[head].isin(counts.index[counts > 1])]
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from
        
    if year_only == True:
        qa_filename = qa_filename + "year_only"
        main_df[constraint_t] = main_df[constraint_t].apply(lambda x: x[-4:])    
        
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
    
    if (ent3 == "count") or (ent3 == "price"):
        triples2[ent3] = triples2[ent3].apply(lambda x: str(cluster_centers[x][2]))
    
    print(triples1)
    print(triples2)

    if ent1 == ent3:
        triples1 = triples1.rename(columns = {ent1 : ent1 + "_a"})
        triples2 = triples2.rename(columns = {ent3 : ent3 + "_c"})
        ent1 = ent1 + "_a"
        ent3 = ent3 + "_c"

    main_df = pd.merge(triples1, triples2, on = ent2)
    print("merged")

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
    
    if (ent3 == "count") or (ent3 == "price"):
        triples2[ent3] = triples2[ent3].apply(lambda x: str(cluster_centers[x][2]))

    if ent1 == ent3:
        triples1 = triples1.rename(columns = {ent1 : ent1 + "_a"})
        triples2 = triples2.rename(columns = {ent3 : ent3 + "_c"})
        ent1 = ent1 + "_a"
        ent3 = ent3 + "_c"

    main_df = pd.merge(triples1, triples2, on = ent2)

    head = ent1
    tail = ent3
    
    if (tail == "count") or (tail == "price"):
        main_df[tail] = main_df[tail].apply(lambda x: str(cluster_centers[x][2]))
    
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








'''
testing = pd.read_csv("triples/funding_round-announced_on-date.csv")

testing["date"] = testing["date"].apply(lambda x: x[-4:])
testing["type"] = testing["date"].apply(lambda x: type(x))

print(testing)
'''
    