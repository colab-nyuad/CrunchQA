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

def pick_answer_1hop(head, rel, tail, constraint, q):
    pass
    
    













    