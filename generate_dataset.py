# -*- coding: utf-8 -*-
import pandas as pd
import random
import pickle
import os
import glob
from string import digits
from datetime import date
from utils import *
import re
import json

# env variables
clustering_data = os.environ['CLUSTERING_DATA']
triples_path = os.environ['TRIPLES_DATA']
dataset_path = os.environ['QA_DATASET']
templates_path = os.environ['TEMPLATES_PATH']
qa_readable_path = os.environ['QA_READABLE']

output_file = "{}/data_advanced.txt".format(dataset_path)
#output_file_numeric = "{}/data_numeric.txt".format(dataset_path)

#clear_content([dataset_path])

# loading clusters
clusters = pickle.load(open("{}/clusters.pickle".format(clustering_data), "rb"))


########################################################################################################
################                 constraints functions
########################################################################################################

''' parameters: main_df: Dataframe, sub_chain: str
    requirement: "subchain: [val1, val2, ..., valn]"
    return: Dataframe
'''
def add_entity_constraint(main_df, constraint):
    chain = list(constraint.keys())[0] # key of the constraint item: sub chain
    values = constraint[chain] # value of the constraint item: specifications
    df_to_join = extract_df2(chain) # sub df of the constraint
    # if there are specifications
    if len(values) > 0:
        for i in range(len(values)):
            if values[i] == "True" or values[i] == "true":
                values[i] = True
            if values[i] == "False"  or values[i] == "false":
                values[i] = False
        condition_col = list(df_to_join.columns)[-1] # column to apply condition is always the last one
        df_to_join = df_to_join[df_to_join[condition_col].isin(values)] # select the sub df with condition column matching the specifications
    join_column = chain.split("-")[0] # column to join is always the first one
    main_df = main_df.merge(df_to_join, on=join_column) # an augmented and selected main_df
    return main_df


''' parameters: main_df: Dataframe, sub_chain: str
    requirement: subchain, values format (before yyyy/mm, after yyyy/mm, between yyyy/mm, yyyy/mm)
    return: an augmented Dataframe
'''
def add_temporal_constraint(main_df, constraint, granularity):
    chain = list(constraint.keys())[0] # key of the constraint item: sub chain
    values = constraint[chain] # value of the constraint item. Format: "between":[date1, date2]  "before":"date"  "after":"date"
    df_to_join = extract_df2(chain) 
    condition_column = df_to_join.columns[-1]
    if len(list(values.keys())) >= 1:
        operator = list(values.keys())[0] # "between", "before", or "after"
        date_val = values[operator] # the specified date(s)

        if isinstance(date_val, list): # if we specify "between"
            end_date_val = date_val[1]
            date_val = date_val[0]
        # set granularity according to specific constraint in the current template
        to_keep = len(str(date_val))
        if to_keep == 4:
            granularity = "year"
        elif to_keep == 7:
            granularity = "month"
        # modify the df_to_join according to the current granularity
        df_to_join[condition_column] = df_to_join[condition_column].apply(lambda x: to_date(x, granularity, inv = True))
        # get the date value of "before" and "after", get the start date value of "between"
        date_val = to_date(date_val, granularity)
        # select by date
        if operator == 'before':
            mask = df_to_join[condition_column].apply(lambda x: x < date_val) == True
        elif operator == 'after':
            mask = df_to_join[condition_column].apply(lambda x: x > date_val) == True
        elif operator == 'between':
            end_date_val = to_date(end_date_val, granularity)
            mask = df_to_join[condition_column].apply(lambda x: end_date_val > x > date_val) == True
        df_to_join = df_to_join[mask]
        # keep as many digits as the specified date range
        df_to_join[condition_column] = df_to_join[condition_column].astype(str)
        df_to_join[condition_column] = df_to_join[condition_column].apply(lambda x: x[:to_keep])
    else:
        df_to_join[condition_column] = df_to_join[condition_column].astype(str)
        # remove day, keep only year and month
        if granularity == "year":
            df_to_join[condition_column] = df_to_join[condition_column].apply(lambda x: x[3:])
    join_column = chain.split("-")[0]
    main_df = main_df.merge(df_to_join, on=join_column)
    return main_df


''' parameters: main_df: Dataframe, constraint: {sub_chain : specification} key-value pair
    Values format: "count_over":"column", "group_by": ["column1", "column2"], "max": "ccolumn"   
    return: an augmented Dataframe
'''

def add_max_constraint(main_df, constraint, main_chain):
    # by default take the maximum of all groups
    chain = list(constraint.keys())[0] # key of the constraint item: sub chain
    # join only if the subchain does not appear in main chain
    reversed_chain = "-".join(chain.split("-")[::-1])
    if chain not in main_chain and reversed_chain not in main_chain:
        df_to_join = extract_df2(chain) 
        join_column = chain.split("-")[0]
        main_df = main_df.merge(df_to_join, on=join_column)
    # value of the constraint item. 
    '''
    Format:  "group_by": "", "max": "column" || "group_by": ["column1", "column2"], "max": "ccolumn"   
    When count_over is "", we don't count, we take the max of what is specified by max
    When count_over is a spedified column, we output the group_wise count to a "count" column
    When group_by is "", we group by the head entity column by default
    When group_by is a list of columns, we group by the columns specifies
    When max is "", we count, group by, and then take the max of a "count" column and take the max according tot he columns to group by
    When max is specified, we directly take the max value of this specified column
    '''
    values = constraint[chain]
    if values["group_by"] == "":
        group_by_cols = [main_df.columns[0]]
    else:
        group_by_cols = values["group_by"]
    if values["count_over"] != "":
        cols_to_use = [i for i in group_by_cols]
        cols_to_use.append(values["count_over"])
        # force all columns involved in grouping and counting into utf-8 format
        for col in cols_to_use:
            main_df[col] = main_df[col].apply(lambda x: x.encode('UTF-8'))
        count_result = main_df.groupby(group_by_cols)[values["count_over"]].count().reset_index(name="count")
        main_df = main_df.merge(count_result, on = group_by_cols)
        # after counting, decode them back
        for col in cols_to_use:
            main_df[col] = main_df[col].apply(lambda x: x.decode('UTF-8'))

    if values["max"] != "":
        max_col = values["max"]
    else:
        max_col = "count"
    main_df = main_df.loc[main_df.reset_index().groupby(group_by_cols)[max_col].idxmax()]
    return main_df


''' parameters: main_df: Dataframe, constraint: {sub_chain : specification} key-value pair
    Values format: "count_over":"column", "group_by": ["column1", "column2"], "numeric": ["ccolumn", ">", 3 ] 
    return: an augmented Dataframe
'''
def add_numeric_constraint(main_df, constraint):
    chain = list(constraint.keys())[0]
    df_to_join = extract_df2(chain)
    # value of the constraint item. 
    '''
    Format:  "group_by": "", "max": "column" || "group_by": ["column1", "column2"], "max": "ccolumn"   
    When count_over is "", we don't count, we take the max of what is specified by max
    When count_over is a spedified column, we output the group_wise count to a "count" column
    When group_by is "", we group by the head entity column by default
    When group_by is a list of columns, we group by the columns specifies
    When numeric is "", we count, group by, and then range select the part where "count" satisfies the numeric specification
    When numeric is specified, directly select the rows with the specified column satisfying the constraint
    '''
    values = constraint[chain]
    join_column = chain.split("-")[0]
    main_df = main_df.merge(df_to_join, on=join_column)
    # group_by_cols are just for count+range selection, not for direct range selection
    if values["group_by"] == "":
        group_by_cols = [main_df.columns[0]]
    else:
        group_by_cols = values["group_by"]
    if values["count_over"] != "":
        cols_to_use = [i for i in group_by_cols]
        cols_to_use.append(values["count_over"])
        # force all columns involved in grouping and counting into utf-8 format
        for col in cols_to_use:
            main_df[col] = main_df[col].apply(lambda x: x.encode('UTF-8'))
        count_result = main_df.groupby(group_by_cols)[values["count_over"]].count().reset_index(name="count")
        main_df = main_df.merge(count_result, on = group_by_cols)
        # after counting, decode them back
        for col in cols_to_use:
            main_df[col] = main_df[col].apply(lambda x: x.decode('UTF-8'))

    if values["numeric"][0] != "":
        numeric_col = values["numeric"][0]
    else:
        numeric_col = "count"
    val = values["numeric"][2]
    # select all rows that satisfies the numeric constraint
    if values["numeric"][1] == "=":
        main_df = main_df[main_df[numeric_col] == val]
    elif values["numeric"][1] == ">=":
        main_df = main_df[main_df[numeric_col] >= val]  
    elif values["numeric"][1] == "<=":
        main_df = main_df[main_df[numeric_col] <= val]  
    elif values["numeric"][1] == ">":
        main_df = main_df[main_df[numeric_col] > val]  
    elif values["numeric"][1] == "<":
        main_df = main_df[main_df[numeric_col] < val]  
    return main_df
 
########################################################################################################

if __name__ == "__main__":
    templates = glob.glob('{}/*.json'.format(templates_path))
    sample_size = 200
    granularity = "year"

    for template in templates[:]:
        template_json = json.load(open(template))
        
        for idx, row in enumerate(template_json['templates']):
            print("-"*70)
            print(row)
            print(template.split('/')[-1], " : ", 'Template {} processed \n'.format(idx), sep = " ")
            columns_to_group_by = []
            main_chain = row['main_chain']
            question = row["question"]
            type = row['type'] if 'type' in row else ''
            columns = main_chain.split('-')
            # head entity
            head = columns[0]
            # tail entity (answer)
            answer = columns[-1]
            # by default group by head
            columns_to_group_by.append(head)
            
            ##### extract the main_df from main chain
            main_df = extract_df2('-'.join(columns[:]))
            
            ##### process constraints
            # iterate through all constraint items of the current template
            for c in row['constraints']:
                # c: a simgle constraint item
                constraint_type = list(c.keys())[0]
                # constraint: key-value pair in the form of {sub_chain: specification}
                constraint = c[constraint_type]
                # produce an updated main_df by augmenting and selecting from main_df according to the constraint
                if constraint_type == 'entity_constraint':
                    main_df = add_entity_constraint(main_df, constraint)
                elif constraint_type == 'temporal_constraint':
                    main_df = add_temporal_constraint(main_df, constraint, granularity)
                elif constraint_type == 'max_constraint':
                    main_df = add_max_constraint(main_df, constraint, main_chain)
                elif constraint_type == 'numeric_constraint':
                    main_df = add_numeric_constraint(main_df, constraint)
                # this is the subchain of the constraint
                chain = list(constraint.keys())[0]
                # add the ending column to the list of columns to group by, because we are reaplacing the values from this column into the question template
                columns_to_group_by.append(chain.split("-")[-1])

            #groupped_df = group_by_question(main_df, columns_to_group_by, answer)      
            #groupped_df['question'] = question
            #groupped_df['type'] = type
            #sampled_df = sample_from_df(groupped_df, sample_size)
            #write_questions(idx, sampled_df, answer, head, output_file, output_file_numeric)
            
            dict_answers_filtered = group_by_question(main_df, columns_to_group_by, answer)
            samples = select_sample(dict_answers_filtered, sample_size)
            write_questions(samples, dict_answers_filtered, question, type, head, output_file)

