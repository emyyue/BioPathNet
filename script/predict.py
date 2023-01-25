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

vocab_file = os.path.join(os.path.dirname(__file__), "../data/QKI/entity_names.txt")
vocab_file = os.path.abspath(vocab_file)


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
def get_prediction(cfg, solver, entity_vocab, relation_vocab):
    test_set = solver.test_set

    dataloader = data.DataLoader(test_set, solver.batch_size, sampler=None, num_workers=solver.num_worker)
    model = solver.model

    model.eval()
    preds = []
    targets = []
    
    for ith, batch in enumerate(dataloader):

        if solver.device.type == "cuda":
            batch = utils.cuda(batch, device=solver.device)
        
        logger.warning("Predicting batch %s" % ith)
        pred, (mask, target) = model.predict_and_target(batch, dataset)
        
        preds.append(pred)
        targets.append(target)
    
    pred = utils.cat(preds)
    target = utils.cat(targets)

    
    return pred, target, mask

def pred_to_dataframe(pred, dataset, entity_vocab, gene_annotation_predict=True):
    # get goterms tail nodes
    nodes = dataset.entity_vocab
    nodes__dict={ix: val for ix, val in enumerate(nodes)}
    go_terms = [val for key, val in nodes__dict.items() if val.startswith('GO:')]
    
    # get head nodes
    testset_nodes = [dataset.entity_vocab[i] for i in [x.numpy()[0] for x in solver.test_set]]
    
    # get both relation and relation^(-1)
    dflist=[]
    for j in [0,1]:
        sigmoid = torch.nn.Sigmoid()
        prob0= sigmoid(pred[:, j, :])
        proball = prob0.flatten().cpu().numpy()

        if gene_annotation_predict:
            df_dict = {'head': np.repeat(testset_nodes, len(go_terms)), 'relation': j,'tail': np.tile(go_terms, len(testset_nodes)), 'probability':proball.tolist()}
        else:
            df_dict = {'head': np.repeat(testset_nodes, len(nodes)), 'relation': j,'tail': np.tile(nodes, len(testset_nodes)), 'probability':proball.tolist()}
            
        dflist.append(df_dict)
    
    #append both dataframes    
    df = pd.DataFrame(dflist[0]).append(pd.DataFrame(dflist[1]))
    
    # fuse with entity vocab given externally
    lookup = pd.DataFrame(list(zip( dataset.entity_vocab, entity_vocab)), columns =['short', 'long'])
        
    df = pd.merge(df, lookup, how="left", left_on="tail", right_on="short")
    
    # sort by head node and probability
    #df = df.sort_values(by=['head', 'probability'],ascending=[True, False])
    return df

        
if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    logger.warning("Config file: %s" % args.config)
    logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)
    
    import pdb; pdb.set_trace()
    
    entity_vocab, relation_vocab = load_vocab(dataset)

    #relation_vocab = ["%s (%d)" % (t[t.rfind("/") + 1:].replace("_", " "), i)
    #                  for i, t in enumerate(dataset.relation_vocab)]
    logger.warning("Starting link prediction")
 
    pred, target, mask= get_prediction(cfg, solver, dataset.entity_vocab, relation_vocab)
    
    df = pred_to_dataframe(pred, dataset, entity_vocab, cfg['task']['gene_annotation_predict']) # TODO: gene_annotation_predict could be nicer
    
    logger.warning("Link prediction done")
    logger.warning("Saving to file")
    df.to_csv(os.path.join( cfg['output_dir'], "results.csv"), index=False)  # TODO:how to save in the dir
    #import pdb; pdb.set_trace()
