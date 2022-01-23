import pandas as pd
import csv
import numpy as np
from collections import Counter
import re
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import math

begin_year = 1900
vanilla_data = 'vanilla'
clustering_data = 'clustering'
regression_data = 'regression'
min_max_scaler = preprocessing.MinMaxScaler()


''' Check uniqueness of series
        parameteres: pandas.series
        return: bool '''
def check_uniqueness(series): 
    return len(series.unique()) == len(series)

''' Merge id and name of the record, replace spaces with '_'
        parameteres: string, string, int
        return: pandas.series '''
def merge_with_id(id_column, name_column, l=5):
    temp_id = id_column.apply(lambda x: x[-l:])
    temp_name = name_column + "@" + temp_id
    return temp_name.apply(lambda x: "_".join(x.split()))

''' Take sample of frame
        parameteres: pandas.dataFrame
        return: pandas.dataFrame '''
def take_sample(data_frame, sample, sample_size):
    if sample:
        return data_frame.loc[:int(sample_size*len(data_frame))]
    return data_frame

''' Select k for k-means using KElbowVisualizer
        parameteres: list'''
def select_k(data):
    kmeanModel = KMeans()
    r = [i for i in range (20, 100, 10)]
    visualizer = KElbowVisualizer(kmeanModel, k=r, timings=False)
    visualizer.fit(data)
    visualizer.show()
    
''' Save statistics for each cluster
        parameteres: list, string'''    
def dump_clusters(kmeans, data, suffix, clusters):
    for i in range(kmeans.n_clusters):
        cluster_data = data[np.where(kmeans.labels_ == i)].ravel()
        min_ = np.min(cluster_data)
        max_ = np.max(cluster_data)
        clusters['cluster-{}-{}'.format(i, suffix)] = [kmeans.cluster_centers_[i].squeeze().tolist(), min_, max_]

def fetch(x, filter_, mapping):
    if not pd.isna(x):
        vals = [v for v in x.split(",") if v in filter_]
        if len(vals) > 0:
            return ','.join([mapping[v.strip()] for v in vals])
    return float('NaN')

def process_appearance_type(row, at):
    if row['appearance_type'] == at:
        return row['event_id']
    return float('NaN')

def fetch_job_title(x, executive_jobs):
    jj = [k for k,v in executive_jobs.items() if any(vv in x for vv in v)]
    if len(jj) > 0:
        return ','.join(jj)
    return float('NaN')

def cluster_data(data, column, clusters, k=40):
    final_kmeans = KMeans(n_clusters=k).fit(data)
    labels = ['cluster-{}-{}'.format(l, column) for l in final_kmeans.labels_]
    mapping = dict(zip(data.squeeze().tolist(), labels))
    dump_clusters(final_kmeans, data, column, clusters)
    return mapping

def encode_dates(series):
    series = series.apply(lambda x: int(x.split('/')[0]) + int(x.split('/')[1]) - begin_year)
    return min_max_scaler.fit_transform(series.values.reshape(-1, 1))
    
def save_clustered_triples(triples):
    triples.to_csv("{}/triples.txt".format(clustering_data), sep='\t', mode='a', index=False, header = None, encoding='utf-8')

def save_regression_triples(triples):
    triples.to_csv("{}/triples.txt".format(regression_data), sep='\t', mode='a', index=False, header = None, encoding='utf-8')

def save_regression_literals(triples):
    triples.to_csv("{}/literals.txt".format(regression_data), sep='\t', mode='a', index=False, header = None, encoding='utf-8')    

def save_vanilla_triples(triples):
    triples.to_csv("{}/triples.txt".format(vanilla_data), sep='\t', mode='a', index=False, header = None, encoding='utf-8')
        
def save_kg_triples(triples):
    save_clustered_triples(triples)
    #save_regression_triples(triples)
    save_vanilla_triples(triples)
    
def save_qa_triples(triples, file_name):
    triples.to_csv("triples/{}.csv".format(file_name), index = False, encoding='utf-8')
    
def create_temporal_triples(data, config_temporal):
    for c in config_temporal:
        c1 = c['c1']
        r = c['rel']
        c2 = c['c2']
        file_name = c['file_name']
        to_drop = c['to_drop']
        
        triples = pd.DataFrame(data[[c1, c2]])
        for td in to_drop:
            triples[c2] = triplets[c2].replace(td, float("NaN"))
        triples = triples.dropna()
        triples[c2] = triples[c2].apply(lambda x:  x[5:7] +'/' + x[:4])
        triples.insert(1, "relation", r)
        triples = triples.drop_duplicates(keep = "last")
                       
        save_clustered_triples(triples)
        save_qa_triples(triples, file_name)
        triples[c2] = encode_dates(triples[c2])
        save_regression_literals(triples)

def create_numerical_triples(data, config_numerical, clusters):
    for c in config_numerical:
        c1 = c['c1']
        r = c['rel']
        c2 = c['c2']
        k = c['k']
        file_name = c['file_name']
        to_drop = c['to_drop']
        
        triples = pd.DataFrame(data[[c1, c2]])
        for td in to_drop:
            triples[c2] = triplets[c2].replace(td, float("NaN"))
        triples = triples.dropna()
        triples.insert(1, "relation", r)
        triples = triples.drop_duplicates(keep = "last")
        save_qa_triples(triples, file_name)
        
        triples_regression = triples.copy()
        
        # Clustering
        clustering_data = np.array(triples[c2].tolist()).reshape((len(triples), 1))
        mapping = cluster_data(clustering_data, c2, clusters, k)
        triples[c2] = triples[c2].map(mapping)    
        save_clustered_triples(triples)
     
        # Regression
        triples_regression[c2] = min_max_scaler.fit_transform(triples_regression[c2].values.reshape(-1, 1))
        save_regression_literals(triples_regression)

def create_regular_triples(data, config_regular, orgs, people, organizations_list, people_list):
    for c in config_regular:
        c1 = c['c1']
        r = c['rel']
        c2 = c['c2']
        file_name = c['file_name']
        to_drop = c['to_drop']
        to_expand = c['to_expand']
        to_explode = c['to_explode']

        triples = pd.DataFrame(data[[c1, c2]])
        for td in to_drop:
            triples[c2] = triples[c2].replace(td, float("NaN"))
        triples = triples.dropna()
        if to_explode == True:
            triples[c2] = triples[c2].apply(lambda x: x.split(","))
            triples = triples.explode(c2)
            triples[c2] = triples[c2].apply(lambda x: x.strip())
            triples[c2] = triples[c2].apply(lambda x: "_".join(x.split()))
        
        triples.insert(1, "relation", r)
        triples = triples.drop_duplicates(keep = "last")
        save_kg_triples(triples)
        save_qa_triples(triples, file_name)
        for e in to_expand:
            if e == 'org': 
                organizations_list.extend(list(set(triples[c1]) & orgs))
            elif e == 'person': 
                people_list.extend(list(set(triples[c1]) & people))
        
def create_triples(data, config, orgs, people, organization_list, people_list, clusters=None):
    for k,v in config.items():
        if k == 'regular':
            create_regular_triples(data, config['regular'], orgs, people, organization_list, people_list)
        elif k == 'numerical':
            create_numerical_triples(data, config['numerical'], clusters)
        elif k == 'temporal':
            create_temporal_triples(data, config['temporal'])
