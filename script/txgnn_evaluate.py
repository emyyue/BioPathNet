import os
import sys
import pprint

import torch

from torchdrug import core
from torchdrug.utils import comm
from torch.utils import data as torch_data
from torchdrug import data, core, utils
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model, task, util#, reasoning_mod
import numpy as np
import pickle


def solver_load(checkpoint, load_optimizer=True):

    if comm.get_rank() == 0:
        logger.warning("Load checkpoint from %s" % checkpoint)
    checkpoint = os.path.expanduser(checkpoint)
    state = torch.load(checkpoint, map_location=solver.device)
    # some issues with loading back the fact_graph and graph
    # remove
    state["model"].pop("fact_graph")
    state["model"].pop("graph")
    state["model"].pop("undirected_fact_graph")
    # load without
    solver.model.load_state_dict(state["model"], strict=False)


    if load_optimizer:
        solver.optimizer.load_state_dict(state["optimizer"])
        for state in solver.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(solver.device)

    comm.synchronize()


def build_solver(cfg):
    cfg.task.model.num_relation = _dataset.num_relation
    _task = core.Configurable.load_config_dict(cfg.task)
    cfg.optimizer.params = _task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    if "scheduler" in cfg:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
    else:
        scheduler = None
    return core.Engine(_task, train_set, valid_set, test_set, optimizer, scheduler, **cfg.engine)

def load_vocab(dataset):
    entity_mapping = {}
    with open(vocab_file, "r") as fin:
        for line in fin:
            k, v = line.strip().split("\t")
            entity_mapping[k] = v
    entity_vocab = [entity_mapping[t] for t in dataset.entity_vocab]
    relation_vocab = ["%s (%d)" % (t[t.rfind("/") + 1:].replace("_", " "), i)
                      for i, t in enumerate(dataset.relation_vocab)]
    
    return entity_vocab, relation_vocab

@torch.no_grad()
def get_prediction(cfg, solver, relation_vocab):
    test_set = solver.test_set

    dataloader = data.DataLoader(test_set, solver.batch_size, sampler=None, num_workers=solver.num_worker)
    model = solver.model

    model.eval()
    preds = []
    targets = []
    masks = []
    
    for ith, batch in enumerate(dataloader):

        if solver.device.type == "cuda":
            batch = utils.cuda(batch, device=solver.device)
        logger.warning("Predicting batch %s" % ith)
        pred, (mask, target) = model.predict_and_target(batch, dataset)
        
        preds.append(pred)
        targets.append(target)
        masks.append(mask)
    
    pred = utils.cat(preds)
    target = utils.cat(targets)
    mask = utils.cat(masks)

    
    return pred, target, mask

def pred_to_dataframe(pred, dataset, entity_vocab, relation_vocab):
    # get head nodes
    
    #testset_nodes = [dataset.entity_vocab[i] for i in [x.numpy()[0] for x in solver.test_set]]
    testset_relation =  [relation_vocab[i] for i in [x.numpy()[2] for x in solver.test_set]]
    nodes = dataset.entity_vocab
    node_type = dataset.graph.node_type
    
    # get both relation and relation^(-1)
    dflist=[]
    for j in [0, 1]:
        # sigmoid = torch.nn.Sigmoid()
        # prob = sigmoid(pred[:, j, :])
        prob = (pred[:, j, :])
        prob = prob.flatten().cpu().numpy()
        df_dict = {'query_node': np.repeat([dataset.entity_vocab[i] for i in [x.numpy()[j] for x in solver.test_set]], len(nodes)),
                   'query_relation': np.repeat(testset_relation, len(nodes)),
                   'reverse': j,
                   'pred_node': np.tile(nodes, len(testset_relation)),
                   'pred_node_type': np.tile(node_type, len(testset_relation)),
                   'probability':prob.tolist()}
         
        # mask out unwanted
        # mymask = temp['pred_node_type'] == 4
        # df_dict = {'query_node': temp['query_node'][mymask],
        #            'query_relation': temp['query_relation'][mymask],
        #            'reverse': j,
        #            'pred_node':  temp['pred_node'][mymask],
        #            'pred_node_type':  temp['pred_node_type'][mymask],
        #            'probability': prob[mymask]}
           
        dflist.append(df_dict)
    df = pd.concat([pd.DataFrame(dflist[0]),pd.DataFrame(dflist[1])])
    lookup = pd.DataFrame(list(zip( dataset.entity_vocab, entity_vocab)), columns =['short', 'long'])

    df = pd.merge(df, lookup, how="left", left_on="query_node", right_on="short", sort=False)
    df = pd.merge(df, lookup, how="left", left_on="pred_node", right_on="short", sort=False)
    return df

def sig(x):
 return 1/(1 + np.exp(-x))


def get_auprc_oneonone(mydir, cfg, df):
    def get_index_of_evalG(rel="contraindication", type="pos", rev=1):
        if rev == 1:
            rel = 'rev_' + rel 
        print("### ", rel, " ###")
        df_g = g.loc[(g.rel == rel) & (g.type==type)]
        x = df_g.merge(diseases, left_on="disease", right_on="idx",how="left")
        y = x.merge(drugs, left_on="drug", right_on="idx",how="left")   
        y.rename(columns={"index_x": "index_disease", "index_y": "index_drug"}, inplace=True)
        #print("Number of missing index or idx: ", y.isnull().sum().sum())
        y['tomatch'] = (y['index_drug'].astype(str) + '_' +  y['index_disease'].astype(str))
        return y

    def get_preds_for_evalG(df_index, rel="contraindication", rev=1):
        df_pred = df.loc[(df.query_relation==rel) & (df.reverse==rev)]
        df_m = df_index.merge(df_pred[['probability','tomatch']], on="tomatch", how="left")
        print("Percentage of missing probabilities replaced by zero: ", df_m['probability'].isnull().sum(), "/", len(df_index))
        df_m["probability"] = df_m["probability"].fillna(0)
        return df_m


    def get_probs(rel="contraindication", type="pos", rev=1):
        df_index = get_index_of_evalG(rel=rel, type=type, rev=rev)
        df_m = get_preds_for_evalG(df_index, rel=rel, rev=rev)
        return torch.tensor(df_m['probability'].values)

    g = pd.read_csv(os.path.join(mydir, "all_data.csv"))
    test = pd.read_csv(os.path.join(cfg.dataset.path, "../test.csv"), sep=",")
    test['x_idx'] = test['x_idx'].astype(int)
    test['y_idx'] = test['y_idx'].astype(int)
    # get mapping for index and idx
    test_dd1 = test.loc[(test.x_type=="disease") & (test.y_type=="drug")]
    test_dd2 = test.loc[(test.x_type=="drug") & (test.y_type=="disease")]

    diseases = pd.concat([test_dd1[['y_index', 'x_idx']].rename(columns={"y_index": "index", "x_idx": "idx"}),
                        test_dd2[['y_index', 'y_idx']].rename(columns={"y_index": "index", "y_idx": "idx"})]).drop_duplicates()

    drugs = pd.concat([test_dd1[['x_index', 'y_idx']].rename(columns={"x_index": "index", "y_idx": "idx"}),
                    test_dd2[['x_index', 'x_idx']].rename(columns={"x_index": "index", "x_idx": "idx"})]).drop_duplicates()
    
    print(drugs.value_counts(['index', 'idx']).sort_values(ascending=False))
    print(diseases.value_counts(['index', 'idx']).sort_values(ascending=False))
    
    pos = {}
    neg = {}

    pos[('drug', 'contraindication', 'disease')] = get_probs(rel="contraindication", type="pos", rev=0)
    pos[('drug', 'indication', 'disease')] = get_probs(rel="indication", type="pos", rev=0)
    pos[('drug', 'off-label use', 'disease')] = get_probs(rel="off-label use", type="pos", rev=0)
    pos[('disease', 'rev_contraindication', 'drug')] = get_probs(rel="contraindication", type="pos", rev=1)
    pos[('disease', 'rev_indication', 'drug')] = get_probs(rel="indication", type="pos", rev=1)
    pos[('disease', 'rev_off-label use', 'drug')] = get_probs(rel="off-label use", type="pos", rev=1)


    neg[('drug', 'contraindication', 'disease')] = get_probs(rel="contraindication", type="neg", rev=0)
    neg[('drug', 'indication', 'disease')] = get_probs(rel="indication", type="neg", rev=0)
    neg[('drug', 'off-label use', 'disease')] = get_probs(rel="off-label use", type="neg", rev=0)
    neg[('disease', 'rev_contraindication', 'drug')] = get_probs(rel="contraindication", type="neg", rev=1)
    neg[('disease', 'rev_indication', 'drug')] = get_probs(rel="indication", type="neg", rev=1)
    neg[('disease', 'rev_off-label use', 'drug')] = get_probs(rel="off-label use", type="neg", rev=1)
    mydict = {'pos' : pos, 'neg': neg}
    return mydict
        
if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)
    print(working_dir)
    vocab_file = os.path.join(os.path.dirname(__file__), cfg.dataset.path, "entity_names.txt")
    vocab_file = os.path.abspath(vocab_file)
    myworkingdir = cfg.output_directory
    mymodel = os.path.split(os.path.split(cfg.checkpoint)[0])[1]
    print(myworkingdir)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
                 
    cfg.dataset.files = ['train1.txt', 'train2.txt', 'valid.txt', 'test_eval.txt']
    _dataset = core.Configurable.load_config_dict(cfg.dataset)
    train_set, valid_set, test_set = _dataset.split()
    
    full_valid_set = valid_set
    if comm.get_rank() == 0:
        logger.warning(_dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    solver = build_solver(cfg)

    if "checkpoint" in cfg:
        solver_load(cfg.checkpoint)
    entity_vocab, relation_vocab = load_vocab(_dataset)

    logger.warning("Starting link prediction")
    pred, target, mask = get_prediction(cfg, solver, relation_vocab)
    df = pred_to_dataframe(pred, _dataset, entity_vocab, relation_vocab)
    df = df.drop_duplicates()
    df['query_relation'] = df['query_relation'].str.split(" \(").str[0]
    #df.to_csv(os.path.join(myworkingdir, mymodel + "_predictions_txgnn.csv"), index=False, sep="\t")
    
    logger.warning("Link prediction done")
    
    logger.warning("Format preds into right dictionary format for TxGNN evaluation - AUPRC")
    mask = (df.reverse == 0)
    df.loc[mask, 'tomatch'] = (df['short_x'].astype(str) + '_' +  df['short_y'].astype(str))
    mask = (df.reverse == 1)
    df.loc[mask, 'tomatch'] = (df['short_y'].astype(str) + '_' +  df['short_x'].astype(str))
    # get preds in dict format
    mydict = get_auprc_oneonone(myworkingdir, cfg, df)
    with open(os.path.join(myworkingdir, mymodel + "g_pos_neg_nbfnet.pickle"), 'wb') as handle: pickle.dump(mydict, handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
    logger.warning("Format preds into right dictionary format for TxGNN evaluation - Recall@k")
    df = df.loc[df.reverse==1]
    df = df.loc[df.pred_node_type==4]
    ##################
    # format node info
    nodes = pd.read_csv(os.path.join(cfg.dataset.path, "entity_all_info.txt"), sep="\t")
    # create lookup dictionaries
    ### for drugs
    drug_map1 = {}
    drug_df = nodes.loc[nodes.node_type=="drug"]
    for i in range(len(drug_df)):
        drug_map1[drug_df.iloc[i,:]['node_name']] = drug_df.iloc[i,:]['node_id']
    ### for diseases
    di_map2 = {}
    di_map1 = {}
    disease_df = nodes.loc[nodes.node_type=="disease"]
    for i in range(len(disease_df)):
        di_map2[disease_df.iloc[i,:]['node_id']] = disease_df.iloc[i,:]['node_name']
        di_map1[disease_df.iloc[i,:]['node_name']] = disease_df.iloc[i,:]['node_id']

    ##################
    for rel in ['contraindication', 'indication', 'off-label use']:
        
        df_rel = df.loc[df.query_relation==rel]
        obj = pd.read_pickle(os.path.join(myworkingdir, "preds_" + rel +".pickle"))        
        # read in dictionary from TxGNN
        goal = obj.copy()
        # get the txgnn prediction dictionaries
        for dis in goal.keys(): # for each disease in disease area split
            goal[dis] = dict.fromkeys(goal[dis], 0)
            if (dis in di_map2):
                df_rel_dis = df_rel.loc[(df_rel.long_x == di_map2[(dis)])]
                print(dis)
                for i in df_rel_dis['long_y']: # for each drug
                    goal[dis][drug_map1[i]] = df_rel_dis.loc[df_rel_dis.long_y == i].iloc[0]['probability']
            elif(dis.split(".")[0] in di_map2):
                dis_m = dis.split(".")[0]
                print(dis_m)
                df_rel_dis = df_rel.loc[(df_rel.long_x == di_map2[(dis_m)])]
                for i in df_rel_dis['long_y']:
                    goal[dis][drug_map1[i]] = df_rel_dis.loc[df_rel_dis.long_y == i].iloc[0]['probability']
            else:
                print(f"not found", dis)
        print(goal[dis][drug_map1[i]])
        print(df_rel_dis.loc[df_rel_dis.long_y == i].iloc[0])
                
        # save
        logger.warning("Save dictionary")
        logger.warning(rel)
        filename = os.path.join(myworkingdir, mymodel + 'preds_' + rel + '_nbfnet.pickle')
        with open(filename, 'wb') as handle: pickle.dump(goal, handle, protocol=pickle.HIGHEST_PROTOCOL)
        