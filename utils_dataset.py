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
#***************************************************************************#
#*************************get and join triplet files************************#
#***************************************************************************#

# get the triplet dataframe corresponding to the head rel and tail
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
    
    record = head + "\t" + rel + "\t" + tail + "\t" + question + "\t" + str(sample_size)
    qa = dict()
    
    main_df = get_df(triples_path, head, rel, tail)
    sample_from = list(set(main_df[head].to_list()))
    
    if len(sample_from) >= sample_size:
        sample = random.sample(sample_from, sample_size)
    else:
        sample = sample_from
        
    for item in sample:
        q = question.replace("[" + head + "]", item)
        a = "|".join(main_df.loc[main_df[head] == item, tail].to_list())
        qa.update({q:a})
        
    qa_df = pd.DataFrame.from_dict(qa, orient = "index")
    qa_df.reset_index(inplace = True)
    qa_df.columns = ["q", "a"]
    qa_df.to_csv("qa/1hop/csv/" + qa_filename + ".csv", index=False, header = None, encoding = "utf-8")
    qa_df.to_csv("qa/1hop/txt/" + qa_filename + ".txt", sep = '\t', index = False, header = None)
    qa_df.to_csv('qa/1hop/qa_1hop.txt', sep = '\t', mode = 'a', index = False, header = None)
    
    
    
        
    
    
    
    
    













    