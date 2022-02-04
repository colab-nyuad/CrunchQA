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

# path for the triplets
triples_path = "triples/"

# path for files that convert non-readables to readables
readable_path = "qa_readable/"

# path for templates
temp_path = "qa_templates/"



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
def get_df(path, h, r, t):
    try:
        filename = path + h + "-" + r + "-" + t + ".csv"
        the_file = pd.read_csv(filename)
    except:
        filename = path + t + "-" + r + "-" + h + ".csv"
        the_file = pd.read_csv(filename)
    return the_file


# path is template path
def pick_answer_1hop(temp_path, triples_path, head, rel, tail, question, sample_size, qa_filename, constraint = None):
    
    
    qa = dict()
    
    # main part of question
    main_df = get_df(triples_path, head, rel, tail)

    # turn unreadables into readables
    if (head == "person") or (head == "org") or (head == "event") or (head == "funding_round") or (head == "fund"):
        main_df[head] = main_df[head].apply(lambda x: readable(x))
    elif (head == "country_code"):
        main_df[head] = main_df[head].apply(lambda x: country_code_to_word[x])
    elif (head == "investment_type"):
        main_df[head] = main_df[head].apply(lambda x: investment_type_to_word[x])
    elif (head == "event_role"):
        main_df[head] = main_df[head].apply(lambda x: " ".join(x.split("_")))
        
    # take a sample
    sample_from = list(set(main_df[head].to_list()))
    random.shuffle(sample_from)
    
    if len(sample_from) >= sample_size:
        sample = random.sample(sample_from, sample_size)
    else:
        sample = sample_from
                
    for item in sample:
        q = question.replace("[" + head + "]", item)
        a = "|".join(list(set(main_df.loc[main_df[head] == item, tail].to_list())))
        qa.update({q:a})
        
    
            
    #print(qa)
    qa_df = pd.DataFrame.from_dict(qa, orient = "index")
    qa_df.reset_index(inplace = True)
    qa_df.columns = ["q", "a"]
    qa_df.to_csv("qa/1hop/csv/" + qa_filename + ".csv", index=False, header = None, encoding = "utf-8")
    qa_df.to_csv("qa/1hop/txt/" + qa_filename + ".txt", sep = '\t', index = False, header = None)
    qa_df.to_csv('qa/1hop/qa_1hop.txt', sep = '\t', mode = 'a', index = False, header = None)
    
    record = head + "\t" + rel + "\t" + tail + "\t" + "no_constraint" + "\t" + question + "\t" + str(len(qa_df)) + "\n"

    with open("qa_templates/qa_dataset_overview.txt", "a") as myfile:
        myfile.write(record)
    
        
    
    
    
def pick_answer_1hop_constraint(temp_path, triples_path, head, rel, tail, question, sample_size, qa_filename, constraint):

    qa = dict()
    
    # main part of question
    main_df = get_df(triples_path, head, rel, tail)
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
    
    # turn unreadables into readables
    if (head == "person") or (head == "org") or (head == "event") or (head == "funding_round") or (head == "fund"):
        main_df[head] = main_df[head].apply(lambda x: readable(x))
    elif (head == "country_code"):
        main_df[head] = main_df[head].apply(lambda x: country_code_to_word[x])
    elif (head == "investment_type"):
        main_df[head] = main_df[head].apply(lambda x: investment_type_to_word[x])
    elif (head == "event_role"):
        main_df[head] = main_df[head].apply(lambda x: " ".join(x.split("_")))
        
    if (constraint_t == "person") or (constraint_t == "org") or (constraint_t == "event") or (constraint_t == "funding_round") or (head == "fund"):
        main_df[constraint_t] = main_df[constraint_t].apply(lambda x: readable(x))
    elif (constraint_t == "country_code"):
        main_df[constraint_t] = main_df[constraint_t].apply(lambda x: country_code_to_word[x])
    elif (constraint_t == "investment_type"):
        main_df[constraint_t] = main_df[constraint_t].apply(lambda x: investment_type_to_word[x])
    elif (constraint_t == "event_role"):
        main_df[constraint_t] = main_df[constraint_t].apply(lambda x: " ".join(x.split("_")))
        
    sample_from = main_df[[head, constraint_t]].drop_duplicates()
    
    if len(sample_from) >= sample_size:
        sample = sample_from.sample(n = sample_size)
    else:
        sample = sample_from
        
    for index, row in sample.iterrows():
        q = question.replace("[" + head + "]", row[head])
        q = q.replace("(" + constraint_t + ")", row[constraint_t])
        mask = (main_df[head] == row[head]) & (main_df[constraint_t] == row[constraint_t])
        result_df = main_df[mask]
        a = "|".join(list(set(result_df[tail].to_list())))
        qa.update({q:a})

    qa_df = pd.DataFrame.from_dict(qa, orient = "index")
    qa_df.reset_index(inplace = True)
    qa_df.columns = ["q", "a"]
    qa_df.to_csv("qa/1hop/csv/" + qa_filename + ".csv", index=False, header = None, encoding = "utf-8")
    qa_df.to_csv("qa/1hop/txt/" + qa_filename + ".txt", sep = '\t', index = False, header = None)
    qa_df.to_csv('qa/1hop/qa_1hop.txt', sep = '\t', mode = 'a', index = False, header = None)
    
    record = head + "\t" + rel + "\t" + tail + "\t" + constraint + "\t" + question + "\t" + str(len(qa_df)) + "\n"

    with open("qa_templates/qa_dataset_overview.txt", "a") as myfile:
        myfile.write(record)












    