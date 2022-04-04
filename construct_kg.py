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
from utils.utils_kg import *
from sklearn import preprocessing
import math
import json
import os, shutil

clusters = {}
organizations_list = []
people_list = []
min_max_scaler = preprocessing.MinMaxScaler()

clear_content()

with open('jobs.json') as jobs_file:
    executive_jobs = json.load(jobs_file)

with open('config.json') as config_file:
    config = json.load(config_file)

#************************************************************************#
#****************** Organizations & People  sets ************************#
#************************************************************************#

org = pd.read_csv("data/organizations.csv")
org = org[org["name"].notna()]
org["name_id"] = merge_with_id(org["uuid"], org["name"])
print("name_id unique? ", check_uniqueness(org["name_id"]))
print("length of dataframe organization:", len(org))

people = pd.read_csv("data/people.csv")
people = people.dropna(subset=['name', 'gender', 'country_code'], how='any')
genders = {"ftm":"transgender_man", "mtf":"transgender_woman", "androgyne":"androgynous"}
people["gender"] = people["gender"].apply(lambda x: genders[x] if x in genders.keys() else x)
print(np.unique(people["gender"]))
people["name_id"] = merge_with_id(people["uuid"], people["name"])
print("name_id unique? ", check_uniqueness(people["name_id"]))
print("length of dataframe people :", len(people))

orgs = set(org['name_id'])
people_names = set(people["name_id"])
orgs_and_people = set()
orgs_and_people = orgs_and_people.union(people_names)
orgs_and_people = orgs_and_people.union(orgs)

#************************************************************************#
#********************************* Jobs *********************************#
#************************************************************************#


config_jobs = config["jobs"]
jobs = pd.read_csv("data/jobs.csv")
jobs = jobs.dropna(subset=['is_current', 'person_name', 'org_name', 'title'], how='any')
jobs["title"] = jobs["title"].apply(lambda x: fetch_job_title(x, executive_jobs))
jobs["person_name"] = merge_with_id(jobs["person_uuid"], jobs["person_name"])
jobs["org_name"] = merge_with_id(jobs["org_uuid"], jobs["org_name"])
jobs["type"] = 'job'
jobs = jobs[jobs['person_name'].isin(people_names)]
jobs = jobs[jobs['org_name'].isin(orgs)]
organizations_list.extend(list(set(jobs["org_name"])))
jobs["uuid"] = jobs["uuid"].apply(lambda x: "jb@" + x[-15:])
print("name_id unique? ", check_uniqueness(jobs["uuid"]))
print("length of dataframe jobs:", len(jobs))
create_triples(jobs, config_jobs, orgs, people_names, organizations_list, people_list)


#************************************************************************#
#********************************* IPOs *********************************#
#************************************************************************#

config_ipos = config["ipos"]
ipos = pd.read_csv("data/ipos.csv")
ipos = ipos.dropna(subset=['org_name'], how='all')
ipos["uuid"] = ipos["uuid"].apply(lambda x: "ipo@" + x[-10:])  # mark it as ipo
ipos["name_id"] = merge_with_id(ipos["org_uuid"], ipos["org_name"])
ipos = ipos.dropna(subset=['share_price_usd', 'money_raised_usd'], how='all')
ipos = ipos[ipos['name_id'].isin(orgs)]
print("name_id unique? ", check_uniqueness(ipos["uuid"]))
print("length of dataframe ipos:", len(ipos))
create_triples(ipos, config_ipos, orgs, people_names, organizations_list, people_list, clusters)


#************************************************************************#
#************************* Organization Parents *************************#
#************************************************************************#

config_organization_parents = config["organization_parents"]
org_pa = pd.read_csv("data/org_parents.csv")
org_pa = org_pa.dropna(subset=['name','parent_name'], how='any')
org_pa["name_id"] = merge_with_id(org_pa["uuid"], org_pa["name"])
org_pa["pname_id"] = merge_with_id(org_pa["parent_uuid"], org_pa["parent_name"])
org_pa = org_pa[org_pa['name_id'].isin(orgs)]
org_pa = org_pa[org_pa['pname_id'].isin(orgs)]
print("length of dataframe organization parents:", len(org_pa))
organizations_list.extend(list(set(org_pa['pname_id'])))
create_triples(org_pa, config_organization_parents, orgs, people_names, organizations_list, people_list)


#************************************************************************#
#******************************* Funding Rounds *************************#
#************************************************************************#

config_funding_rounds = config["funding_rounds"]
fr = pd.read_csv("data/funding_rounds.csv")
fr = fr.dropna(subset=['raised_amount_usd', 'org_name'], how='any')
fr["org_name_id"] = merge_with_id(fr["org_uuid"], fr["org_name"])
fr["name_id"] = merge_with_id(fr["uuid"], fr["name"])
fr = fr.drop_duplicates(subset = "name_id", keep = "last")
fr = fr[fr['org_name_id'].isin(orgs)]
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
save_vanilla_triples(triples)
save_qa_triples(triples, 'investment_type-subtype_of-type_funding_round')


#************************************************************************#
#************************** Investment Partners *************************#
#************************************************************************#

config_investment_partners = config["investment_partners"]
ip = pd.read_csv("data/investment_partners.csv")
ip = ip.dropna(subset=['investor_name', 'partner_name'], how='any')
ip["iname_id"] = merge_with_id(ip["investor_uuid"], ip["investor_name"])
ip["pname_id"] = merge_with_id(ip["partner_uuid"], ip["partner_name"])
ip["fund_id"] = merge_with_id(ip["funding_round_uuid"], ip["funding_round_name"])
ip = ip[ip['iname_id'].isin(orgs_and_people)]
ip = ip[ip['pname_id'].isin(orgs_and_people)]
print("length of dataframe investment partners:", len(ip))
create_triples(ip, config_investment_partners, orgs, people_names, organizations_list, people_list) 


#************************************************************************#
#********************************** Investors  **************************#
#************************************************************************#

config_investors = config["investors"]
investors = pd.read_csv("data/investors.csv")
investors = investors[investors["name"].notna()]
investors["name_id"] = merge_with_id(investors["uuid"], investors["name"])
investors = investors.dropna(subset=['total_funding_usd', 'investment_count'], how='all')
investors = investors[investors['name_id'].isin(orgs_and_people)]
investors["investor_types"] = investors["investor_types"].replace(float("NaN"), "investor")
print("name_id unique? ", check_uniqueness(investors["name_id"]))
print("length of dataframe investors:", len(investors))

investors["investor_types"] = investors["investor_types"].apply(lambda x: x.split(","))
investors = investors.explode("investor_types")
#investors["investor_types"] = investors["investor_types"].replace('angel', "angel_investor")
ivsubtype = list(investors.investor_types.unique())
ivsubtype.remove("investor")
triples = pd.DataFrame(ivsubtype)
triples.insert(1, "relation", "subtype_of")
triples.insert(2, "type", "investor")
triples.columns = ["investor_type", "relation", "type_investor"]
save_vanilla_triples(triples)
save_qa_triples(triples, 'investor_type-subtype_of-type_investor')
create_triples(investors, config_investors, orgs, people_names, organizations_list, people_list, clusters)  


#************************************************************************#
#********************************** Funds *******************************#
#************************************************************************#

config_funds = config["funds"]
funds = pd.read_csv("data/funds.csv")
funds = funds.dropna(subset=['name', 'entity_name'], how='any')
funds["name_id"] = merge_with_id(funds["uuid"], funds["name"])
funds["ename_id"] = merge_with_id(funds["entity_uuid"], funds["entity_name"])
funds = funds.dropna(subset=['raised_amount_usd'], how='all')
funds = funds[funds['ename_id'].isin(orgs)]
funds['type'] = 'fund'
print("name_id unique? ", check_uniqueness(funds['name_id']))
print("length of dataframe funds:", len(funds))
create_triples(funds, config_funds, orgs, people_names, organizations_list, people_list, clusters)   


#************************************************************************#
#*************************** Acquisitions *******************************#
#************************************************************************#

config_acquisitions = config["acquisitions"]
a = pd.read_csv("data/acquisitions.csv")
a = a.dropna(subset=['acquirer_name', 'acquiree_name'], how='any')
a["uuid"] = a["uuid"].apply(lambda x: "ac@" + x[-10:])
a["rname_id"] = merge_with_id(a["acquirer_uuid"],a["acquirer_name"])
a["ename_id"] = merge_with_id(a["acquiree_uuid"],a["acquiree_name"])
a = a[a['rname_id'].isin(orgs)]
a = a[a['ename_id'].isin(orgs)]
a["acquisition_type"] = a["acquisition_type"].replace(float("NaN"), "acquisition")
print("name_id unique? ", check_uniqueness(a["uuid"]))
print("length of dataframe acquisitions:", len(a))   

triples = list(a.acquisition_type.unique())
triples.remove("acquisition")
triples = pd.DataFrame(triples)
triples.insert(1, "relation", "subtype_of")
triples.insert(2, "type_acquisition","acquisition")
triples.columns = ["acquisition_type", "relation", "type_acquisition"]
save_vanilla_triples(triples)
save_qa_triples(triples, 'acquisition_type-subtype_of-type_acquisition')
create_triples(a, config_acquisitions, orgs, people_names, organizations_list, people_list, clusters)    

#************************************************************************#
#********************************* Events *******************************#
#************************************************************************#

config_events = config["events"]
e = pd.read_csv("data/events.csv")
e = e[e["name"].notna()]
e["name_id"] = merge_with_id(e["uuid"],e["name"])
print("name_id unique? ", check_uniqueness(e["name_id"]))
print("length of dataframe events:", len(e))
e["event_roles"] = e["event_roles"].apply(lambda x: x.split(","))
e = e.explode("event_roles")
e["event_roles"] = e["event_roles"].replace("other", float("NaN"))
e["event_roles"] = e["event_roles"].fillna('event')
triples = list(e["event_roles"].unique())
triples.remove('event')
triples = pd.DataFrame(triples)
print(triples)
triples.insert(1, "relation", "subtype_of")
triples.insert(2, "type", "event")
triples.columns = ["event_role", "relation", "type_event"]
save_vanilla_triples(triples)
save_qa_triples(triples, 'event_role-subtype_of-type_event')
create_triples(e, config_events, orgs, people_names, organizations_list, people_list)

#************************************************************************#
#************************ Event_appearance ******************************#
#************************************************************************#

config_event_appearance = config["event_appearance"]
ea = pd.read_csv("data/event_appearances.csv")
ea = ea.dropna(subset=['participant_name', 'event_name'], how='any')
ea["name_id"] = merge_with_id(ea["participant_uuid"],ea["participant_name"])
ea["event_id"] = merge_with_id(ea["event_uuid"],ea["event_name"])

print("name_id unique? ", check_uniqueness(ea["event_id"]))
print("length of dataframe event appearance:", len(ea))
appearance_types = ea['appearance_type'].unique()
for at in appearance_types:
    ea[at] = ea.apply(lambda row: process_appearance_type(row, at), axis=1)

create_triples(ea, config_event_appearance, orgs, people_names, organizations_list, people_list)    

#************************************************************************#
#************************ Organizations *********************************#
#************************************************************************#

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
