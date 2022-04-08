import pandas as pd
import random
import pickle
import os
import glob
from string import digits
from datetime import date
from utils import *
import re

# env variables
triples_path = "triples/"
templates_path = "qa_templates/"
qa_readable_path = "qa_readable/"
dataset_path = "qa_dataset/"
output_file = "{}/data.txt".format(dataset_path)
output_file_numeric = "{}/data_numeric.txt".format(dataset_path)
sample_size = 10

# clear previous contents
#clear_content([dataset_path])

# loading clusters
clusters = pickle.load(open("kg/clustering/clusters.pickle", "rb"))

# countries dict
ccode_df = pd.read_csv(qa_readable_path + "country_code.csv")
ccode_dict = dict(zip(ccode_df['country_code'], ccode_df['name']))

# investment type dict
inv_type = pd.read_csv(qa_readable_path + "investment_type.csv")
inv_dict = dict(zip(inv_type['investment_type'], inv_type['name']))

def format_entity(entity):
    entity = str(entity)
    entity = entity.strip()
    if entity in ccode_dict.keys():
        entity = ccode_dict[entity]
    elif entity in inv_dict.keys():
        entity = inv_dict[entity]
    if "@" in entity:
        entity = entity.split("@")[0]
    if "_" in entity:
        entity = " ".join(entity.split("_"))
    return entity

def to_date(x):
    return date(int(x.split("/")[1]), int(x.split("/")[0]), 1)

def extract_df(file_name):
    col_names = file_name.split("-")
    col_names_without_loc = [re.sub(r'\d+', '', s) for s in col_names]
    file_name = '{}/{}.csv'.format(triples_path, "-".join(col_names_without_loc))

    if os.path.isfile(file_name):
        df = pd.read_csv(file_name)
    else:
        # in case filename is inverted, we read it and swap columns back
        file_name = '{}/{}-{}-{}.csv'.format(triples_path, col_names_without_loc[2], col_names_without_loc[1], col_names_without_loc[0])
        df = pd.read_csv(file_name)
        columns_titles = df.columns  # put column in the same order defined in file_name
        columns_titles = [columns_titles[2],columns_titles[1],columns_titles[0]]
        df=df.reindex(columns=columns_titles)

    col_names[1] = "relation"
    df.columns = col_names  # add location digits back into the file name

    return df


''' parameters: main_df: Dataframe, sub_chain: str
    requirement: "subchain: [val1, val2, ..., valn]"
    return: Dataframe'''
def add_simple_constraint(main_df, sub_chain):
    if ":" not in sub_chain:
        df_to_join = extract_df(sub_chain)
    else:
        values = sub_chain.split(":")[1].strip()[1:-1]
        sub_chain = sub_chain.split(":")[0].strip()
        df_to_join = extract_df(sub_chain)
        values = [s.strip() for s in values.split(",")]
        condition_col = list(df_to_join.columns)[-1]
        df_to_join = df_to_join[df_to_join[condition_col].isin(values)]

    join_column = sub_chain.split("-")[0]
    main_df = main_df.merge(df_to_join, on=join_column)
    return main_df


''' parameters: main_df: Dataframe, sub_chain: str
    requirement: subchain: before yyyy/mm
                           after yyyy/mm 
                           between yyyy/mm, yyyy/mm
    return: an augmented Dataframe
'''
def add_temporal_constraint(main_df, sub_chain):
    if ":" not in sub_chain:
        df_to_join = extract_df(sub_chain)
    else:
        sub_chain, condition = sub_chain.split(":")
        print(sub_chain, condition)
        df_to_join = extract_df(sub_chain.strip())
        condition_column = df_to_join.columns[-1]
        condition = condition.strip()
        if ',' in condition:
            operator, vals, end_vals = condition.split(' ')
            vals = vals[:-1]
        else:
            operator, vals = condition.split(' ')
        operator = operator.strip()

        year, month = vals.strip().split("/")
        date_val = date(int(year), int(month), 1)
        df_to_join[condition_column] = df_to_join[condition_column].apply(lambda x: to_date(x))
        
        if operator == 'before':
            mask = df_to_join[condition_column].apply(lambda x: x < date_val) == True
        elif operator == 'after':
            mask = df_to_join[condition_column].apply(lambda x: x > date_val) == True
        elif operator == 'between':
            year, month = end_vals.strip().split("/")
            end_date_val = date(int(year), int(month), 1)
            mask = df_to_join[condition_column].apply(lambda x: end_date_val > x > date_val) == True
        
        df_to_join = df_to_join[mask]
        df_to_join[condition_column] = df_to_join[condition_column].astype(str)
        df_to_join[condition_column] = df_to_join[condition_column].apply(lambda x: x[:-3])

    join_column = sub_chain.split("-")[0]
    main_df = main_df.merge(df_to_join, on=join_column)
    return main_df
    

''' parameters: main_df: Dataframe, sub_chain: str
    requirement: entity1, entity2
    return: an augmented Dataframe
'''
def add_aggregation_max_constraint(main_df, sub_chain):
    # by default take the maximum of all groups
    if ":" not in sub_chain:
        groupby_cols = list(main_df.columns[::2])
    else:
        sub_chain, cols = sub_chain.split(":")
        groupby_cols = cols.strip().split(",")
        sub_chain = sub_chain.strip()

    df_to_join = extract_df(sub_chain)
    max_col = df_to_join.columns[-1]
    join_column = sub_chain.split("-")[0]
    main_df = main_df.merge(df_to_join, on=join_column)
    main_df = main_df.loc[main_df.reset_index().groupby(groupby_cols)[max_col].idxmax()]
    
    return main_df

'''description'''
def group_by_question(df, columns_to_group_by, answer_column):
    df[answer_column] = df.groupby(columns_to_group_by)[answer_column].transform(lambda x: ' || '.join(x))
    return df.drop_duplicates()

'''description'''
def sample_from_df(sample_from, sample_size):
    if len(sample_from) <= sample_size:
        sample_size = len(sample_from)
    return sample_from.sample(n = sample_size)

'''description'''
def substitute_entities(row, head_column):
    columns_to_substitute = re.findall( r'\((.*)\)', row['question'])
    for column in columns_to_substitute:
        row['question'] = row['question'].replace('(' + column + ')', format_entity(row[column]))

    # Replace topic entity
    row['question'] = row['question'].replace('[' + head_column + ']', format_entity(row[head_column]))
    return row['question']

'''description'''
def write_questions(sampled_df, answer_column, head_column, output_file):
    sampled_df['question'] = sampled_df.apply(lambda x: substitute_entities(x, head_column), axis=1)
    if sampled_df.dtypes[answer_column] is str:
        sampled_df[answer_column] = sampled_df[answer_column].apply(lambda x: ' || '.join([format_entity(ans) for ans in x.split(' || ')]))
    questions = sampled_df['question']
    answers = sampled_df[answer_column]
    heads = sampled_df[head_column]
    types = sampled_df['type']

    with open(output_file, 'a', encoding = "utf-8") as fout, open(output_file_numeric, 'a', encoding = "utf-8") as fout_numeric:
        for q, h, ans, type in zip(questions, heads, answers, types):
            if pd.isna(type):
                line = '{}\t{}\t{}\n'.format(q, h, ans)
                fout.write(line)
            else:
                line = '{}\t{}\t{}\t{}\n'.format(q, h, ans, type)
                fout_numeric.write(line)


if __name__ == "__main__":
    templates = glob.glob('{}*.csv'.format(templates_path))    
    for template in templates:
        template_df = pd.read_csv(template, encoding = "utf-8")
        for index, row in template_df.iterrows():
            columns_to_group_by = []
            main_chain = row['main_chain']
            question = row['question']
            type = row['type']
            columns = main_chain.split('-')
            head = columns[0]
            answer = columns[-1]
            columns_to_group_by.append(head)

            # extract the starting part of the main chain
            main_df = extract_df('-'.join(columns[:3]))

            # connect rest of the main chain to the main_df
            for i in range(3, len(columns), 2):
                sub_chain = '-'.join([columns[i-1], columns[i], columns[i+1]])
                df_to_join = extract_df(sub_chain)
                main_df = main_df.merge(df_to_join, on=columns[i-1])

            # process constraints
            simple_constraint = row["simple_constraint"]
            if isinstance(simple_constraint, str):
                for sub_chain in simple_constraint.split("|"):
                    main_df = add_simple_constraint(main_df, sub_chain)
                    columns_to_group_by.append(sub_chain.split(":")[0].strip().split("-")[-1].strip())

            temporal_constraint = row["temporal_constraint"]
            if isinstance(temporal_constraint, str):
                for sub_chain in temporal_constraint.split("|"):
                    main_df = add_temporal_constraint(main_df, sub_chain)
                    columns_to_group_by.append(sub_chain.split(":")[0].strip().split("-")[-1].strip())

            aggregation_max_constraint = row["aggregation_max_constraint"]
            if isinstance(aggregation_max_constraint, str):
                for sub_chain in aggregation_max_constraint.split("|"):
                    main_df = add_aggregation_max_constraint(main_df, sub_chain)
                    columns_to_group_by.append(sub_chain.split(":")[0].strip().split("-")[-1].strip())

            aggregation_sum_constraint = row["aggregation_sum_constraint"]
            if isinstance(aggregation_sum_constraint, str):
                for sub_chain in aggregation_sum_constraint.split("|"):
                    main_df = add_aggregation_sum_constraint(main_df, sub_chain)
                    columns_to_group_by.append(sub_chain.split(":")[0].strip().split("-")[-1].strip())

            aggregation_count_entity_constraint = row["aggregation_count_entity_constraint"]
            if isinstance(aggregation_count_entity_constraint, str):
                for sub_chain in aggregation_count_entity_constraint.split("|"):
                    main_df = aggregation_count_entity_constraint(main_df, sub_chain)
                    columns_to_group_by.append(sub_chain.split(":")[0].strip().split("-")[-1].strip())

            main_df['question'] = question
            main_df['type'] = type
            groupped_df = group_by_question(main_df, columns_to_group_by, answer)      
            sampled_df = sample_from_df(groupped_df, sample_size)
            write_questions(sampled_df, answer, head, output_file)

            print(template.strip("qa_templates\\template_").strip(".csv"), "|", 'Template {} processed \n'.format(index), sep = " ")







