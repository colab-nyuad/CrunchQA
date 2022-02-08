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
from utils import *
from sklearn import preprocessing
import math
import json
import os, shutil

clusters = {}
organizations_list = []
people_list = []
sample = False
sample_size = 0.02
begin_year = 1900
n_clusters = 40
vanilla_data = 'vanilla'
clustering_data = 'clustering'
regression_data = 'regression'
min_max_scaler = preprocessing.MinMaxScaler()


# clear previous contents
folders = ["vanilla/", "clustering/", "regression/", "triples/"]
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
print("----------all previous kg files cleared----------")


executive_jobs = {'president' : ['president'],
                  'VP' : ['vice president', 'vp'],
                  'founder' : ['founding partner', 'founder', 'co-founder', 'co founder', 'cofounder'],
                  'owner' : ['owner'],
                  'advisor' : ['advisor'],
                  'director' : ['director'],
                  'COO' : ['chief operating officer', 'coo'],
                  'CFO' : ['chief financial officer', 'cfo'], 
                  'CEO' : ['chief executive officer', 'ceo'],
                  'CRO' : ['chief revenue officer', 'cro'],
                  'CPO' : ['chief people officer', 'cpo'],
                  'CSO' : ['chief security officer', 'cso'], 
                  'CBO' : ['chief business officer', 'cbo'],
                  'CTO' : ['chief technology offcier', 'cto'],
                  'CIO' : ['chief information officer', 'cio'],
                  'board_member' : ['chairman','boar director','board member']}

with open('config.json') as config_file:
    config = json.load(config_file)

#************************************************************************#
#****************** Organizations & People  sets ************************#
#************************************************************************#

org = pd.read_csv("data/organizations.csv")
org = org[org["name"].notna()]
org["name_id"] = merge_with_id(org["uuid"], org["name"])
org["name"] = org["name"].apply(lambda x: "".join(x.split("|")))
org = take_sample(org, sample, sample_size)
print("name_id unique? ", check_uniqueness(org["name_id"]))
print("length of dataframe organization:", len(org))

people = pd.read_csv("data/people.csv")
people = people.dropna(subset=['name', 'gender', 'country_code'], how='any')
people["name"] = people["name"].apply(lambda x: "".join(x.split("|")))
people["name_id"] = merge_with_id(people["uuid"], people["name"])
people = take_sample(people, sample, sample_size)
print("name_id unique? ", check_uniqueness(people["name_id"]))
print("length of dataframe people :", len(people))

'''
event_appearances = pd.read_csv("data/event_appearances.csv")
event_appearance_df = event_appearances[["participant_name", "participant_uuid", "participant_type"]]
event_appearance_df = event_appearance_df.dropna()

event_appearance_people = event_appearance_df[event_appearance_df["participant_type"] == "person"]
event_appearance_people["appearance_people_name_id"] = merge_with_id(event_appearance_people["participant_uuid"], event_appearance_people["participant_name"])
event_appearance_people = take_sample(event_appearance_people, sample, sample_size)

event_appearance_org = event_appearance_df[event_appearance_df["participant_type"] == "organization"]
event_appearance_org["appearance_org_name_id"] = merge_with_id(event_appearance_org["participant_uuid"], event_appearance_org["participant_name"])
event_appearance_org = take_sample(event_appearance_org, sample, sample_size)

investor_df = pd.read_csv("data/investors.csv")
investor_df = investor_df[["uuid", "name", "type"]]
investor_df = investor_df.dropna()
investor_people = investor_df[investor_df["type"] == "person"]
investor_people["investor_people_name_id"] = merge_with_id(investor_people["uuid"], investor_people["name"])
investor_people = take_sample(investor_people, sample, sample_size)

investor_org = investor_df[investor_df["type"] == "organization"]
investor_org["investor_org_name_id"] = merge_with_id(investor_org["uuid"], investor_org["name"])
investor_org = take_sample(investor_org, sample, sample_size)
'''
orgs = set(org['name_id'])
'''
orgs = orgs.union(set(event_appearance_org["appearance_org_name_id"]))
orgs = orgs.union(set(investor_org["investor_org_name_id"]))git add 
'''

people_names = set(people["name_id"])
'''
people_names = people_names.union(set(event_appearance_people["appearance_people_name_id"]))
people_names = people_names.union(set(investor_people["investor_people_name_id"]))
'''

orgs_and_people = set()
orgs_and_people = orgs_and_people.union(people_names)
orgs_and_people = orgs_and_people.union(orgs)


#print(list(people_names)[:20])
#print(list(orgs)[:20])

#************************************************************************#
#********************************* Jobs *********************************#
#************************************************************************#
# checked
# tested

config_jobs = config["jobs"]
jobs = pd.read_csv("data/jobs.csv")
jobs = jobs.dropna(subset=['is_current', 'person_name', 'org_name', 'title'], how='any')
jobs["title"] = jobs["title"].apply(lambda x: "".join(x.split("|")))
jobs["title"] = jobs["title"].apply(lambda x: fetch_job_title(x, executive_jobs))
jobs["person_name"] = jobs["person_name"].apply(lambda x: "".join(x.split("|")))
jobs["person_name"] = merge_with_id(jobs["person_uuid"], jobs["person_name"])
jobs["org_name"] = jobs["org_name"].apply(lambda x: "".join(x.split("|")))
jobs["org_name"] = merge_with_id(jobs["org_uuid"], jobs["org_name"])
jobs["type"] = 'job'
jobs = jobs[jobs['person_name'].isin(people_names)]
jobs = jobs[jobs['org_name'].isin(orgs)]
organizations_list.extend(list(set(jobs["org_name"])))
jobs["uuid"] = jobs["uuid"].apply(lambda x: "jb@" + x[-15:])
jobs = take_sample(jobs, sample, sample_size)
print("name_id unique? ", check_uniqueness(jobs["uuid"]))
print("length of dataframe jobs:", len(jobs))
create_triples(jobs, config_jobs, orgs, people_names, organizations_list, people_list)


#************************************************************************#
#********************************* IPOs *********************************#
#************************************************************************#
# checked
# tested

config_ipos = config["ipos"]
ipos = pd.read_csv("data/ipos.csv")
ipos = ipos.dropna(subset=['org_name'], how='all')
ipos["uuid"] = ipos["uuid"].apply(lambda x: "ipo@" + x[-10:])  # mark it as ipo
ipos["org_name"] = ipos["org_name"].apply(lambda x: "".join(x.split("|")))
ipos["name_id"] = merge_with_id(ipos["org_uuid"], ipos["org_name"])
ipos = ipos.dropna(subset=['share_price_usd', 'money_raised_usd'], how='all')
ipos = ipos[ipos['name_id'].isin(orgs)]
ipos = take_sample(ipos, sample, sample_size)
print("name_id unique? ", check_uniqueness(ipos["uuid"]))
print("length of dataframe ipos:", len(ipos))
create_triples(ipos, config_ipos, orgs, people_names, organizations_list, people_list, clusters)


#************************************************************************#
#************************* Organization Parents *************************#
#************************************************************************#
#
# tested

config_organization_parents = config["organization_parents"]
org_pa = pd.read_csv("data/org_parents.csv")
org_pa = org_pa.dropna(subset=['name','parent_name'], how='any')
org_pa["name"] = org_pa["name"].apply(lambda x: "".join(x.split("|")))
org_pa["name_id"] = merge_with_id(org_pa["uuid"], org_pa["name"])
org_pa["parent_name"] = org_pa["parent_name"].apply(lambda x: "".join(x.split("|")))
org_pa["pname_id"] = merge_with_id(org_pa["parent_uuid"], org_pa["parent_name"])
org_pa = org_pa[org_pa['name_id'].isin(orgs)]
org_pa = org_pa[org_pa['pname_id'].isin(orgs)]
org_pa = take_sample(org_pa, sample, sample_size)
print("length of dataframe organization parents:", len(org_pa))
organizations_list.extend(list(set(org_pa['pname_id'])))
create_triples(org_pa, config_organization_parents, orgs, people_names, organizations_list, people_list)


#************************************************************************#
#******************************* Funding Rounds *************************#
#************************************************************************#
# checked
# tested

config_funding_rounds = config["funding_rounds"]
fr = pd.read_csv("data/funding_rounds.csv")
fr = fr.dropna(subset=['raised_amount_usd', 'org_name'], how='any')
fr["org_name"] = fr["org_name"].apply(lambda x: "".join(x.split("|")))
fr["org_name_id"] = merge_with_id(fr["org_uuid"], fr["org_name"])
fr["name"] = fr["name"].apply(lambda x: "".join(x.split("|")))
fr["name_id"] = merge_with_id(fr["uuid"], fr["name"])
fr = fr.drop_duplicates(subset = "name_id", keep = "last")
fr = fr[fr['org_name_id'].isin(orgs)]
fr = take_sample(fr, sample, sample_size)
print("name_id unique? ", check_uniqueness(fr["name_id"]))
print("length of dataframe funding rounds:", len(fr))

filter_ = set(org.uuid).union(set(people.uuid))
orgd = dict(zip(org.uuid, org.name_id))
perd = dict(zip(people.uuid, people.name_id))
mapping = {**orgd, **perd}
fr["lead_investor_uuids"] = fr["lead_investor_uuids"].apply(lambda x: fetch(x, filter_, mapping))
create_triples(fr, config_funding_rounds, orgs, people_names, organizations_list, people_list, clusters)

triples = pd.DataFrame(fr["investment_type"].unique())
triples.insert(1, "relation", "subtype_of")
triples.insert(2, "type", "funding_round")
triples.columns = ["investment_type", "subtype_of", "type_funding_round"]
save_kg_triples(triples)
save_qa_triples(triples, 'investment_type-subtype_of-type_funding_round')


#************************************************************************#
#************************** Investment Partners *************************#
#************************************************************************#
#
# tested

config_investment_partners = config["investment_partners"]
ip = pd.read_csv("data/investment_partners.csv")
ip = ip.dropna(subset=['investor_name', 'partner_name'], how='any')
ip["investor_name"] = ip["investor_name"].apply(lambda x: "".join(x.split("|")))
ip["iname_id"] = merge_with_id(ip["investor_uuid"], ip["investor_name"])
ip["partner_name"] = ip["partner_name"].apply(lambda x: "".join(x.split("|")))
ip["pname_id"] = merge_with_id(ip["partner_uuid"], ip["partner_name"])
ip["funding_round_name"] = ip["funding_round_name"].apply(lambda x: "".join(x.split("|")))
ip["fund_id"] = merge_with_id(ip["funding_round_uuid"], ip["funding_round_name"])
ip = ip[ip['iname_id'].isin(orgs_and_people)]
ip = ip[ip['pname_id'].isin(orgs_and_people)]
ip = take_sample(ip, sample, sample_size)
print("length of dataframe investment partners:", len(ip))
create_triples(ip, config_investment_partners, orgs, people_names, organizations_list, people_list) 


#************************************************************************#
#********************************** Investors  **************************#
#************************************************************************#
# checked
# tested

config_investors = config["investors"]
investors = pd.read_csv("data/investors.csv")
investors = investors[investors["name"].notna()]
investors["name"] = investors["name"].apply(lambda x: "".join(x.split("|")))
investors["name_id"] = merge_with_id(investors["uuid"], investors["name"])
investors = investors.dropna(subset=['total_funding_usd', 'investment_count'], how='all')
investors = investors[investors['name_id'].isin(orgs_and_people)]
investors["investor_types"] = investors["investor_types"].replace(float("NaN"), "investor")
investors = take_sample(investors, sample,sample_size)
print("name_id unique? ", check_uniqueness(investors["name_id"]))
print("length of dataframe investors:", len(investors))

investors["investor_types"] = investors["investor_types"].apply(lambda x: x.split(","))
investors = investors.explode("investor_types")
investors["investor_types"] = investors["investor_types"].replace('angel', "angel_investor")
ivsubtype = list(investors.investor_types.unique())
ivsubtype.remove("investor")
triples = pd.DataFrame(ivsubtype)
triples.insert(1, "relation", "subtype_of")
triples.insert(2, "type", "investor")
triples.columns = ["investor_type", "relation", "type_investor"]
save_kg_triples(triples)
save_qa_triples(triples, 'investor_type-subtype_of-type_investor')
create_triples(investors, config_investors, orgs, people_names, organizations_list, people_list, clusters)  


#************************************************************************#
#********************************** Funds *******************************#
#************************************************************************#
# checked
# tested

config_funds = config["funds"]
funds = pd.read_csv("data/funds.csv")
funds = funds.dropna(subset=['name', 'entity_name'], how='any')
funds["name"] = funds["name"].apply(lambda x: "".join(x.split("|")))
funds["name_id"] = merge_with_id(funds["uuid"], funds["name"])
funds["entity_name"] = funds["entity_name"].apply(lambda x: "".join(x.split("|")))
funds["ename_id"] = merge_with_id(funds["entity_uuid"], funds["entity_name"])
funds = funds.dropna(subset=['raised_amount_usd'], how='all')
funds = funds[funds['ename_id'].isin(orgs)]
funds['type'] = 'fund'
funds = take_sample(funds, sample, sample_size)
print("name_id unique? ", check_uniqueness(funds['name_id']))
print("length of dataframe funds:", len(funds))
create_triples(funds, config_funds, orgs, people_names, organizations_list, people_list, clusters)   


#************************************************************************#
#*************************** Acquisitions *******************************#
#************************************************************************#
# checked

config_acquisitions = config["acquisitions"]
a = pd.read_csv("data/acquisitions.csv")
a = a.dropna(subset=['acquirer_name', 'acquiree_name'], how='any')
a["uuid"] = a["uuid"].apply(lambda x: "ac@" + x[-10:])
a["acquirer_name"] = a["acquirer_name"].apply(lambda x: "".join(x.split("|")))
a["rname_id"] = merge_with_id(a["acquirer_uuid"],a["acquirer_name"])
a["acquiree_name"] = a["acquiree_name"].apply(lambda x: "".join(x.split("|")))
a["ename_id"] = merge_with_id(a["acquiree_uuid"],a["acquiree_name"])
a = a[a['rname_id'].isin(orgs)]
a = a[a['ename_id'].isin(orgs)]
a["acquisition_type"] = a["acquisition_type"].replace(float("NaN"), "acquisition")
a = take_sample(a, sample, sample_size)
print("name_id unique? ", check_uniqueness(a["uuid"]))
print("length of dataframe acquisitions:", len(a))   

triples = list(a.acquisition_type.unique())
triples.remove("acquisition")
if len(triples) > 0:
    triples = pd.DataFrame(triples)
    triples.insert(1, "relation", "subtype_of")
    triples.insert(2, "type_acquisition","acquisition")
    triples.columns = ["acquisition_type", "relation", "type_acquisition"]
    save_kg_triples(triples)
    save_qa_triples(triples, 'acquisition_type-subtype_of-type_acquisition')
create_triples(a, config_acquisitions, orgs, people_names, organizations_list, people_list, clusters)    

#************************************************************************#
#********************************* Events *******************************#
#************************************************************************#
# checked
# tested

config_events = config["events"]
e = pd.read_csv("data/events.csv")
e = e[e["name"].notna()]
e["name"] = e["name"].apply(lambda x: "".join(x.split("|")))
e["name_id"] = merge_with_id(e["uuid"],e["name"])
e = take_sample(e, sample, sample_size)
print("name_id unique? ", check_uniqueness(e["name_id"]))
print("length of dataframe events:", len(e))
e["event_roles"] = e["event_roles"].apply(lambda x: x.split(","))
e = e.explode("event_roles")
e["event_roles"] = e["event_roles"].replace("other", float("NaN"))
triples = list(e["event_roles"].unique())
triples = pd.DataFrame(triples)
triples.insert(1, "relation", "subtype_of")
triples.insert(2, "type", "event")
triples.columns = ["event_role", "relation", "type_event"]
save_kg_triples(triples)
save_qa_triples(triples, 'event_role-subtype_of-type_event')
create_triples(e, config_events, orgs, people_names, organizations_list, people_list)    

#print("event countrycode: ", e.country_code.unique())

#************************************************************************#
#************************ Event_appearance ******************************#
#************************************************************************#
# checked
# tested

config_event_appearance = config["event_appearance"]
ea = pd.read_csv("data/event_appearances.csv")
ea = ea.dropna(subset=['participant_name', 'event_name'], how='any')
ea["participant_name"] = ea["participant_name"].apply(lambda x: "".join(x.split("|")))
ea["name_id"] = merge_with_id(ea["participant_uuid"],ea["participant_name"])
ea["event_name"] = ea["event_name"].apply(lambda x: "".join(x.split("|")))
ea["event_id"] = merge_with_id(ea["event_uuid"],ea["event_name"])

ea = take_sample(ea, sample, sample_size)
print("name_id unique? ", check_uniqueness(ea["event_id"]))
print("length of dataframe event appearance:", len(ea))
appearance_types = ea['appearance_type'].unique()
for at in appearance_types:
    ea[at] = ea.apply(lambda row: process_appearance_type(row, at), axis=1)

#print(ea)
#print(ea.sponsor.unique())
#print(ea.speaker.unique())
#print(ea.organizer.unique())
#print(ea.contestant.unique())
#print(ea.exhibitor.unique())
create_triples(ea, config_event_appearance, orgs, people_names, organizations_list, people_list)    

#************************************************************************#
#************************ Organizations *********************************#
#************************************************************************#
# 
# tested

config_organizations = config["organizations"]
organizations_list = set(organizations_list)
org = org[org['name_id'].isin(organizations_list)]
org['type'] = 'organization'
create_triples(org, config_organizations, orgs, people_names, organizations_list, people_list)     


#************************************************************************#
#******************************* People *********************************#
#************************************************************************#

config_people = config["people"]
people_list = set(people_list)
people = people[people['name_id'].isin(people_list)]
people['type'] = 'person'
create_triples(people, config_people, orgs, people_names, organizations_list, people_list)


#*********** Dumping clusters statistics ****************#
with open('{}/clusters.pickle'.format(clustering_data), 'wb') as f:
    pickle.dump(clusters, f)
