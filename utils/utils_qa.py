# -*- coding: utf-8 -*-
import re
import os
import pandas as pd 
import random
from datetime import date


qa_readable_path = os.environ['QA_READABLE']
triples_path = os.environ['TRIPLES_DATA']

# countries dict
ccode_df = pd.read_csv(qa_readable_path + "/country_code.csv", encoding='utf-8')
ccode_dict = dict(zip(ccode_df['country_code'], ccode_df['name']))

# investment type dict
inv_type = pd.read_csv(qa_readable_path + "/investment_type.csv", encoding='utf-8')
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

def to_date(x, granularity, inv=False):
    if granularity == "year":
        year_idx = 1 if inv else 0
        return date(int(x.split("/")[year_idx]), 1, 1)
    elif granularity == "month":
        year_idx = 1 if inv else 0
        month_idx = 0 if inv else 1
        return date(int(x.split("/")[year_idx]), int(x.split("/")[month_idx]), 1)

# extract multi-hop dataframe
def extract_df2(main_chain):
    columns = main_chain.split("-")
    main_df = extract_df('-'.join(columns[:3]))
    for i in range(3, len(columns), 2):
        sub_chain = '-'.join([columns[i-1], columns[i], columns[i+1]])
        df_to_join = extract_df(sub_chain)
        main_df = main_df.merge(df_to_join, on=columns[i-1])
    return main_df

# extract 1hop dataframe
def extract_df(main_chain):
    if "/" not in main_chain:
        return extract_df_helper(main_chain)
    else:
    # multi-entity/multi-relation filename, for example person/org-country_code-country_code
        col_names = main_chain.split("-")
        for s in col_names:
            if "/" in s:
                multi_column = s
                single_columns = multi_column.split("/")

        result_df = pd.DataFrame(columns=[col_names[0], 'relation', col_names[2]])

        for col in single_columns:
            main_chain_col = main_chain.replace(multi_column, col)
            df = extract_df_helper(main_chain_col)
            df.columns = result_df.columns
            result_df = pd.concat([result_df, df], ignore_index=True)
        return result_df

def extract_df_helper(main_chain):
    col_names = main_chain.split("-")
    # remove location number
    col_names_without_loc = [re.sub(r'\d+', '', s) for s in col_names]
    file_name = '{}/{}.csv'.format(triples_path, "-".join(col_names_without_loc))
    if os.path.isfile(file_name):
        df = pd.read_csv(file_name, encoding='utf-8-sig')
    else:
    # in case filename is inverted, we read it and swap columns back
        file_name = '{}/{}-{}-{}.csv'.format(triples_path, col_names_without_loc[2], col_names_without_loc[1], col_names_without_loc[0])
        df = pd.read_csv(file_name, encoding='utf-8-sig')
        df=df.reindex(columns=[df.columns[2], df.columns[1], df.columns[0]])
    col_names[1] = "relation"
    df.columns = col_names
    return df

def group_by_question(df, columns_to_group_by, answer_column, limit=20):
    df.columns = [x.encode('utf-8').decode('utf-8', 'ignore') for x in df.columns]
    columns_involved = []
    for c in columns_to_group_by:
        columns_involved.append(c)
    if answer_column not in columns_involved:
        columns_involved.append(answer_column)
    df = df[columns_involved].drop_duplicates()
    df[answer_column] = df[answer_column].apply(lambda x: str(x))
    dict_answers = {}
    for index, row in df.iterrows():
        vals = row[columns_to_group_by]
        col_values = tuple([(index, value) for index, value in vals.iteritems()])
        if col_values in dict_answers:
            dict_answers[col_values].append(row[answer_column])
        else:
            dict_answers[col_values] = [row[answer_column]]
    dict_answers_filtered = {k: v for k,v in dict_answers.items() if len(v) < limit}
    return dict_answers_filtered
    #df[answer_column] = df.groupby(columns_to_group_by)[answer_column].transform(lambda x: ' || '.join(x))
    #df = df.drop_duplicates()
    #return df

'''
def sample_from_df(sample_from, sample_size):
    if len(sample_from) <= sample_size:
        sample_size = len(sample_from)
    return sample_from.sample(n = sample_size)


def substitute_entities(row, head_column):
    columns_to_substitute = re.findall( r'\((.*?)\)', row['question'])

    for column in columns_to_substitute:
        row['question'] = row['question'].replace('(' + column + ')', format_entity(row[column]))
    row['question'] = row['question'].replace('[' + head_column + ']', format_entity(row[head_column]))
    return row['question']


def write_questions(id, sampled_df, answer_column, head_column, output_file, output_file_numeric):
    sampled_df['question'] = sampled_df.apply(lambda x: substitute_entities(x, head_column), axis=1)
    if sampled_df.dtypes[answer_column] is str:
        sampled_df[answer_column] = sampled_df[answer_column].apply(lambda x: ' || '.join([format_entity(ans) for ans in x.split(' || ')]))
    questions = sampled_df['question']
    answers = sampled_df[answer_column]
    heads = sampled_df[head_column]
    types = sampled_df['type']
    with open(output_file, 'a', encoding = "utf-8") as fout, open(output_file_numeric, 'a', encoding = "utf-8") as fout_numeric:
        for q, h, ans, type in zip(questions, heads, answers, types):
            if type == '':
                line = '{}\t{}\t{}\n'.format(q, h, ans)
                fout.write(line)
            else:
                line = '{}\t{}\t{}\t{}\n'.format(q, h, ans, type)
                fout_numeric.write(line)
'''

def select_sample(dict_answers_filtered, sample_size):
    if len(dict_answers_filtered) < sample_size:
        sample_size = len(dict_answers_filtered)
    samples = random.sample(list(dict_answers_filtered), sample_size)
    return samples


def substitute_entities(sample, head_column, question):
    columns_to_substitute = re.findall( r'\((.*?)\)', question)
    sample = {ss[0]:ss[1] for ss in sample}
    for column in columns_to_substitute:
        question = question.replace('(' + column + ')', format_entity(sample[column]))
    question = question.replace('[' + head_column + ']', format_entity(sample[head_column]))
    return sample[head_column], question

def write_questions(samples, dict_answers_filtered, question, type, head_column, output_file):
    if isinstance(question, list):
        paraphrase = True
    else:
        paraphrase = False
    with open(output_file, 'a', encoding = "utf-8") as fout:
        for s in samples:
            # randomly pick a question pharaphrase for questions with paraphrases
            if paraphrase == True:
                head, q = substitute_entities(s, head_column, random.choice(question))
            else:
                head, q = substitute_entities(s, head_column, question)
            ans = ' || '.join(dict_answers_filtered[s])
            if type == '':
                line = '{}\t{}\t{}\n'.format(q, head, ans)
            else:
                line = '{}\t{}\t{}\t{}\n'.format(q, head, ans, type)
            print(line)
            fout.write(line)
