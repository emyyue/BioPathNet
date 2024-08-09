import pandas as pd
import numpy as np
import os

## Read in data

# cd ./reproduce/synleth/

# Define threshold and directory name
THR = 0.90
DIR_NAME = "KR4SL_thr0{}".format(str(THR).replace('.', '').lstrip('0'))

# Define input and output directories
input_dir = f"./{DIR_NAME}/data/transductive/original/"
output_dir = "./data/transductive_noRev/"
os.makedirs(output_dir, exist_ok=True)

# Processed data from KR4SL
facts = pd.read_csv(os.path.join(input_dir, "facts.txt"), sep=" ", names=['h', 'r', 't'])
entities = pd.read_csv(os.path.join(input_dir, "entities.txt"), sep=" ", names=['node', 'index', 'type'])
train = pd.read_csv(os.path.join(input_dir, "train_filtered.txt"), sep=" ", names=['h', 'r', 't'])
valid = pd.read_csv(os.path.join(input_dir, "valid_filtered.txt"), sep=" ", names=['h', 'r', 't'])
test = pd.read_csv(os.path.join(input_dir, "test_filtered.txt"), sep=" ", names=['h', 'r', 't'])
facts['split'] = "train1" 
train['split'] = "train2" 
valid['split'] = "valid" 
test['split'] = "test"
data = pd.concat([train, valid, test])
all = pd.concat([data, facts])
all['ht'] = all['h'] + '_' + all['t']
print("all.head")
print(all.head())
print(all.value_counts('split'))

# Data from SynLethDB 
# dowloaded (https://synlethdb.sist.shanghaitech.edu.cn/v2/static/download/SL/Human_SL.csv) and put in ./data
raw = pd.read_csv("./data/raw_SynLethDB/Human_SL.csv")
raw[raw['r.statistic_score'] > 0.1]

## Remove duplicates from the original data
facts = pd.read_csv(os.path.join(input_dir, "facts.txt"), sep=" ", names=['h', 'r', 't'])
entities = pd.read_csv(os.path.join(input_dir, "entities.txt"), sep=" ", names=['node', 'index', 'type'])
train = pd.read_csv(os.path.join(input_dir, "train_filtered.txt"), sep=" ", names=['h', 'r', 't'])
valid = pd.read_csv(os.path.join(input_dir, "valid_filtered.txt"), sep=" ", names=['h', 'r', 't'])
test = pd.read_csv(os.path.join(input_dir, "test_filtered.txt"), sep=" ", names=['h', 'r', 't'])
facts['split'] = "train1" 
train['split'] = "train2" 
valid['split'] = "valid" 
test['split'] = "test"
data = pd.concat([train, valid, test])
all = pd.concat([data, facts])
all['ht'] = all['h'] + '_' + all['t']
print("Remove duplicates from the original data")
print(all.head())
print(all.shape)
# Sort according to h and t; find the duplicates
all_sort = all.copy()
all_sort[['x', 'y']] = np.sort(all_sort[['h', 't']], axis=1)
all_sort['dup'] = all_sort.duplicated(['x', 'y'])
print("all_sort")
print(all_sort)
# Sort according to h and t AND r; find the duplicates
all_sort2 = all.copy()
all_sort2[['x', 'y', 'z']] = np.sort(all_sort[['h', 't', 'r']], axis=1)
all_sort2['dup'] = all_sort2.duplicated(['x', 'y', 'z'])
print("all_sort2")
print(all_sort2)
print(all_sort2.loc[all_sort.dup==True].value_counts('split'))
print(all_sort2.loc[all_sort.dup==True].value_counts('split'))
# Actually drop the duplicates and call the dataframe all_noRev
all_sort = all.copy()
all_sort[['x', 'y', 'z']] = np.sort(all_sort[['h', 't', 'r']], axis=1)
all_sort['dup'] = all_sort.duplicated(['x', 'y', 'z'])
all_noRev = all_sort.drop_duplicates(['x', 'y', 'z']).drop(columns=['x', 'y', 'z'], axis=1)
all_sort[['x', 'y']] = np.sort(all_sort[['h', 't']], axis=1)
all_sort.duplicated(['x', 'y'])
all_sort['dup'] = all_sort.duplicated(['x', 'y'])
print(np.sort(all_sort[['h', 't', 'r']], axis=1))
print(all_sort.value_counts('dup'))
print(all_noRev.value_counts('dup'))
print(all_noRev)


########## Filter data

filtered = raw[raw['r.statistic_score'] < THR]

todrop_1 = filtered['n1.name'] + '_' + filtered['n2.name']
todrop_2 = filtered['n2.name'] + '_' + filtered['n1.name']
all_noRev['filter_1'] = np.where(all_noRev.ht.isin(todrop_1), True, False)
all_noRev['filter_2'] = np.where(all_noRev.ht.isin(todrop_2), True, False)

print(all_noRev.value_counts('filter_1'))

print("############# Edges that are removed #############")
print(all_noRev[all_noRev['filter_1']].value_counts('split'))
print("##### Edges that are kept ############# ")
print(all_noRev[~all_noRev['filter_1']].value_counts('split'))

all_noRev_filt = all_noRev[~all_noRev['filter_1']]

## Extract data for BioPathNet - remove reverse AND those with low threshold

print(all_noRev_filt.value_counts('split'))

test_eval = pd.DataFrame({"h" : pd.concat([all_noRev_filt.loc[all_noRev_filt.split=="test"]['h'], all_noRev_filt.loc[all_noRev_filt.split=="test"]['t']]).unique(), 
             "r" : "SL_GsG",
             "t": pd.concat([all_noRev_filt.loc[all_noRev_filt.split=="test"]['h'], all_noRev_filt.loc[all_noRev_filt.split=="test"]['t']]).unique()[::-1]})
             
test_eval.to_csv(os.path.join(output_dir, "test_pred.txt"), index=False, header=False, sep="\t")
all_noRev_filt.loc[all_noRev_filt.split == "train1"][['h', 'r', 't']].to_csv(os.path.join(output_dir, "train1.txt"), index=False, header=False, sep="\t")
all_noRev_filt.loc[all_noRev_filt.split == "train2"][['h', 'r', 't']].to_csv(os.path.join(output_dir, "train2.txt"), index=False, header=False, sep="\t")
all_noRev_filt.loc[all_noRev_filt.split == "valid"][['h', 'r', 't']].to_csv(os.path.join(output_dir, "valid.txt"), index=False, header=False, sep="\t")
all_noRev_filt.loc[all_noRev_filt.split == "test"][['h', 'r', 't']].to_csv(os.path.join(output_dir, "test.txt"), index=False, header=False, sep="\t")

# get test_pred
df = all_noRev_filt.loc[all_noRev_filt.split=="test"][['h', 'r', 't']]
entities.sort_values(by=['type'], inplace=True)
entities['type_id'] =  entities['type'].astype('category').cat.codes
entities[['node', 'node']].to_csv(os.path.join(output_dir, "entity_names.txt"), index=False, header=False, sep="\t")
entities[['node', 'type_id']].to_csv(os.path.join(output_dir, "entity_types.txt"), index=False, header=False, sep="\t")


## Extract data for KR4SL             

filtered = raw[raw['r.statistic_score'] < THR]

todrop_1 = filtered['n1.name'] + '_' + filtered['n2.name']
todrop_2 = filtered['n2.name'] + '_' + filtered['n1.name']
all['filter_1'] = np.where(all.ht.isin(todrop_1), True, False)
all['filter_2'] = np.where(all.ht.isin(todrop_2), True, False)

filter = all['filter_1'] | all['filter_2']
all_filt = all[~filter]

all_filt.loc[all_filt.split == "train1"][['h', 'r', 't']].to_csv(os.path.join(input_dir, "facts.txt"), index=False, header=False, sep="\t")
all_filt.loc[all_filt.split == "train2"][['h', 'r', 't']].to_csv(os.path.join(input_dir, "train_filtered.txt"), index=False, header=False, sep="\t")
all_filt.loc[all_filt.split == "valid"][['h', 'r', 't']].to_csv(os.path.join(input_dir, "valid_filtered.txt"), index=False, header=False, sep="\t")
all_filt.loc[all_filt.split == "test"][['h', 'r', 't']].to_csv(os.path.join(input_dir, "test_filtered.txt"), index=False, header=False, sep="\t")