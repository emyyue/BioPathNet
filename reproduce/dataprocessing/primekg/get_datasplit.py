import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import os
import pandas as pd
import torch
import numpy as np
from txgnn import TxData



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed for random split")
    parser.add_argument("--split", help="which data split desired; possible are ['random', 'complex_disease', 'disease_eval', 'cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland', 'full_graph', 'downstream_pred']")
    parser.add_argument("--source", help="Source location")
    parser.add_argument("--dest", help="Destination location")
    return parser.parse_args()





def get_txgnn_datasplit(seed, split, source):
    print('\n############################################')
    print('Process data with TxGNN...\n')
    data = TxData(data_folder_path = source)
    data.prepare_split(split = split, seed = seed)
    
    ######### get specific test data
    # preprocessing was done in TxGNN on rev relations --> extract those 
    # revert the rev relations back
    # and extract the x_index and y_index (which do not need to be flipped)
    test_indication = data.df_test[data.df_test.relation == 'rev_indication'].replace('rev_indication', 'indication')
    test_contra = data.df_test[data.df_test.relation == 'rev_contraindication'].replace('rev_contraindication', 'contraindication')
    test_off = data.df_test[data.df_test.relation == 'rev_off-label use'].replace('rev_off-label use', 'off-label use')
    # but x_type and y_type needs to be flipped
    test_indication[['x_type','y_type']] = test_indication[['y_type','x_type']]
    test_contra[['x_type','y_type']] = test_contra[['y_type','x_type']]
    test_off[['x_type','y_type']] = test_off[['y_type','x_type']]
    ######### others
    df_valid = data.df_valid
    df_train = data.df_train
    df_test = data.df_test
    df = data.df
    G = data.G
    return df_train, df_valid, df_test, df, G, test_indication, test_contra, test_off

def remove_rev_relations(df_train, df_valid):
    print('\n############################################')
    print('Generating train1, train2, valid for NBFNet...\n')
    print("Remove reverse relations...")
    df_valid = df_valid[~df_valid['relation'].str.startswith("rev_")]
    df_train = df_train[~df_train['relation'].str.startswith("rev_")]
    print("Drug - disease relations only for loss...")
    interested = ["off-label use", "indication", "contraindication"]
    train_bool = df_train['relation'].isin(interested)
    valid_bool = df_valid['relation'].isin(interested)
    df_train_brg = df_train[~train_bool]
    df_valid_brg = df_valid[~valid_bool]
    df_train_dd = df_train[train_bool]
    df_valid_dd = df_valid[valid_bool]
    print("Assemble train1, train2, valid for NBFNet...")
    train1 = pd.concat([df_train_brg, df_valid_brg]).sample(frac = 1)
    train2 = df_train_dd.sample(frac = 1)
    valid = df_valid_dd.sample(frac = 1)
    return train1, train2, valid

def get_unique_nodes(df):
    dt = pd.concat([
        df[['x_index', 'x_type']].rename(columns={"x_index": "idx", "x_type": "type"}),
        df[['y_index', 'y_type']].rename(columns={"y_index": "idx", "y_type": "type"})]).drop_duplicates()
    return dt

def get_statistics(train1, train2, valid, test_indication, test_contra, test_off, entities, split):
    print("##################\nRelations...\n")
    print("\ttrain1")
    print(train1['relation'].value_counts())
    print("\ttrain2")
    print(train2['relation'].value_counts())
    print("\tvalid")
    print(valid['relation'].value_counts())
    print("\ttest - indication")
    print(test_indication.value_counts(['x_type', 'relation', 'y_type']))
    print("\ttest - contra")
    print(test_contra.value_counts(['x_type', 'relation', 'y_type']))
    print("\ttest - off")
    print(test_off.value_counts(['x_type', 'relation', 'y_type']))
    print(test_indication.head())
    print("\n##################\nNodes...\n")
    print("\ttrain1")
    print(get_unique_nodes(train1)['type'].value_counts())
    print("\ttrain2")
    print(get_unique_nodes(train2)['type'].value_counts())
    print("\tvalid")
    print(get_unique_nodes(valid)['type'].value_counts())
    test = pd.concat([test_indication, test_contra, test_off])
    print("\ttest - all")
    print(get_unique_nodes(test)['type'].value_counts())
    print("\ttest - indication")
    print(get_unique_nodes(test_indication)['type'].value_counts())
    print("\ttest - contra")
    print(get_unique_nodes(test_contra)['type'].value_counts())
    print("\ttest - off")
    print(get_unique_nodes(test_off)['type'].value_counts())
    print("\n##################\nShapes:")
    dt_all = pd.concat([train1, train2, valid, test_indication, test_contra, test_off])
    print(f"\tShape of train1 ", train1.shape)
    print(f"\tShape of train2 ", train2.shape)
    print(f"\tShape of valid ", valid.shape)
    print(f"\tShape of test - indication ", test_indication.shape)
    print(f"\tShape of test - contraindication ", test_contra.shape)
    print(f"\tShape of test - off-label use", test_off.shape)
    print(f"\tShape of all", dt_all.shape)
    print("\n##################\nNodes not connected in the graph:")
    test = pd.concat([test_indication, test_contra, test_off])
    train_x = pd.concat([train1, train2])['x_index'].unique()
    train_y = pd.concat([train1, train2])['y_index'].unique()
    print(f"\tvalid: num of x (drugs) not in fact_graph", sum(~valid['x_index'].drop_duplicates().isin(train_x)))
    print(f"\tvalid: num of y (diseases) not in fact_graph", sum(~valid['y_index'].drop_duplicates().isin(train_y)))
    print("Number of edges affected of valid - drug: \n", valid[['x_type','x_index', 'y_type', 'y_index']][~valid['x_index'].isin(train_x)])
    print("Number of edges affected of valid - disease: \n", valid[['x_type','x_index', 'y_type', 'y_index']][~valid['y_index'].isin(train_y)])
    print(f"\ttest: num of x (drugs) not in fact_graph", sum(~test['x_index'].drop_duplicates().isin(train_x)))
    print(f"\ttest: num of y (diseases) not in fact_graph", sum(~test['y_index'].drop_duplicates().isin(train_y)))
    # be careful that the x_id and y_id are reverted though
    print("Number of edges affected of test - drug: \n", test[['x_type','x_index', 'y_type', 'y_index']][~test['x_index'].isin(train_x)])
    print("Number of edges affected of test - disease: \n", test[['x_type','x_index', 'y_type', 'y_index']][~test['y_index'].isin(train_y)])
    entities.loc[entities.node_index.isin(test[~test['y_index'].isin(train_y)].y_index)]
    df = test.merge(entities, left_on="y_index", right_on="idx")
    dis = pd.read_csv(os.path.join("../data/disease_files/", split + ".csv"))
    df2 =  df.merge(dis, on="node_name", how="left")[['y_index', 'node_name', 'node_type_y']].drop_duplicates()
    print("Number of diseases not in diseses files: ", df2['node_type_y'].isnull().sum())
    print(df2.loc[df2.node_type_y.isnull()])
    print("Done")

def save_data(source, split, seed, train1, train2, valid, test_indication, test_contra, test_off):
    print('\n############################################')
    print('Saving data to...')
    x = split +  "_" +  str(seed)
    path = os.path.join(source, x, "nfbnet/")

    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print(f"The new directory", path, " is created!")
    
    print(os.path.join(path, "train1.txt"))
    print(os.path.join(path, "train2.txt"))
    print(os.path.join(path, "valid.txt"))
    print(os.path.join(path, "test_indi.txt"))
    print(os.path.join(path, "test_contra.txt"))
    print(os.path.join(path, "test_off.txt"))
    test = pd.concat([test_indication, test_contra, test_off])
    # for prediction of each disease with every drug, query only needed once
    test_y = test.drop_duplicates(subset=['relation', 'y_index'], keep='first')
    test_x = test.drop_duplicates(subset=['relation', 'x_index'], keep='first')
    test_eval = pd.concat([test_y, test_x]).drop_duplicates()
    print("num of unique drugs: ", len(test_eval['x_index'].unique()))
    print("num of unique diseases: ", len(test_eval['y_index'].unique()))
    
   
    train1[['x_index', 'relation', 'y_index']].to_csv(os.path.join(path, "train1.txt"), sep='\t', index=False, header=False)
    train2[['x_index', 'relation', 'y_index']].to_csv(os.path.join(path,"train2.txt"), sep='\t', index=False, header=False)
    valid[['x_index', 'relation', 'y_index']].to_csv(os.path.join(path,"valid.txt"), sep='\t', index=False, header=False)
    test[['x_index', 'relation', 'y_index']].to_csv(os.path.join(path,"test.txt"), sep='\t', index=False, header=False)
    test_eval[['x_index', 'relation', 'y_index']].to_csv(os.path.join(path,"test_eval_full.txt"), sep='\t', index=False, header=False)
    test_y[['x_index', 'relation', 'y_index']].to_csv(os.path.join(path,"test_eval.txt"), sep='\t', index=False, header=False)
    test_indication[['x_index', 'relation', 'y_index']].to_csv(os.path.join(path,"test_indi.txt"), sep='\t', index=False, header=False)
    test_contra[['x_index', 'relation', 'y_index']].to_csv(os.path.join(path,"test_contra.txt"), sep='\t', index=False, header=False)
    test_off[['x_index', 'relation', 'y_index']].to_csv(os.path.join(path,"test_off.txt"), sep='\t', index=False, header=False)
    print("Done")
    
    
def get_entity_info(train1, train2, valid, test_indication, test_contra, test_off, source, split, seed):
    print('\n############################################')
    print('Info about entities and types...')
    test = pd.concat([test_indication, test_contra, test_off])
    df_all = pd.concat([train1, train2, valid, test])
    nodes  = pd.read_csv(os.path.join(source, "nodes.csv"), sep=',')
    
    entities = get_unique_nodes(df_all)
    print(f"Number of unique nodes: ", len(entities.idx.unique()))
    print(f"Number of unique types: ", len(entities.type.unique()))

    entities = entities.merge(nodes, how='left', left_on='idx', right_on='node_index')
    entities.sort_values(by=['type'], inplace=True)
    entities['type_id'] =  entities['type'].astype('category').cat.codes
    
    print(entities.head())
    x = split +  "_" +  str(seed)
    path = os.path.join(source, x, "nfbnet/")
    # get different test splits for indication, contraindication
    # only of diseases of interest
    entities[['idx', 'type_id']].to_csv(os.path.join(path, "entity_types.txt"), sep='\t', index=False, header=False)
    entities[['idx', 'node_name']].to_csv(os.path.join(path, "entity_names.txt"), sep='\t', index=False, header=False)
    entities[['idx','node_id', 'node_name', 'node_type', 'type_id', 'node_source']].to_csv(os.path.join(path, "entity_all_info.txt"), sep='\t', index=False)
    return entities


if __name__ == "__main__":
    args = parse_args()
    seed = int(args.seed)
    split = str(args.split)
    source = args.source
    torch.manual_seed(seed)
    np.random.seed(seed)    
    df_train, df_valid, df_test, df, G, test_indication, test_contra, test_off = get_txgnn_datasplit(seed=seed, split=split, source=source)
    train1, train2, valid = remove_rev_relations(df_train, df_valid)
    save_data(source, split, seed, train1, train2, valid, test_indication, test_contra, test_off)
    entities = get_entity_info(train1, train2, valid, test_indication, test_contra, test_off, source, split, seed)
    get_statistics(train1, train2, valid, test_indication, test_contra, test_off, entities, split)


