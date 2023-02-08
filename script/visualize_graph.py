import os
import re
import pprint
import shutil
import logging
import sys
import argparse
from collections import defaultdict

import torch
from torch.utils import data as torch_data

import torchdrug
from torchdrug import core, data
from torchdrug.utils import comm, plot


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model, task, util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file",
                        default="config/knowledge_graph/wn18rr.yaml")
    parser.add_argument("-s", "--start", help="start config id for hyperparmeter search", type=int,
                        default=None)
    parser.add_argument("-e", "--end", help="end config id for hyperparmeter search", type=int,
                        default=None)

    return parser.parse_known_args()[0]


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


def get_freebase_vocab(_dataset, freebase2wikidata="~/kg-datasets/freebase2wikidata.txt",
                       wikidata_alias="~/kg-datasets/wikidata5m_entity.txt"):
    entity_fb2wiki = {}
    freebase2wikidata = os.path.expanduser(freebase2wikidata)
    with open(freebase2wikidata, "r") as fin:
        for i in range(4):
            fin.readline()
        for line in fin:
            tokens = line.strip().split("\t")
            freebase_id = re.search(r"<http://rdf.freebase.com/ns(.+)>", tokens[0]).groups()[0]
            freebase_id = freebase_id.replace(".", "/")
            wikidata_id = re.search(r"<http://www.wikidata.org/entity/(.+)>", tokens[2]).groups()[0]
            entity_fb2wiki[freebase_id] = wikidata_id

    entity2alias = {}
    wikidata_alias = os.path.expanduser(wikidata_alias)
    with open(wikidata_alias, "r") as fin:
        for line in fin:
            tokens = line.strip().split("\t")
            entity_id = tokens[0]
            alias = tokens[1]
            entity2alias[entity_id] = "%s (%s)" % (alias, entity_id)

    entity_vocab = _dataset.entity_vocab
    new_entity_vocab = {}
    for i, token in enumerate(entity_vocab):
        if token in entity_fb2wiki and entity_fb2wiki[token] in entity2alias:
            new_entity_vocab[i] = entity2alias[entity_fb2wiki[token]]
        elif token in entity_fb2wiki:
            new_entity_vocab[i] = entity_fb2wiki[token]
        else:
            new_entity_vocab[i] = token

    relation_vocab = _dataset.relation_vocab
    new_relation_vocab = {i: "%s (%d)" % (token[token.rfind("/") + 1:].replace("_", " "), i)
                          for i, token in enumerate(relation_vocab)}

    return new_entity_vocab, new_relation_vocab


def visualize_echarts(graph, sample, paths, weights, entity_vocab, relation_vocab, ranking=None, save_file=None):
    triplet2id = {tuple(edge.tolist()): i for i, edge in enumerate(graph.edge_list)}
    edge_weight = defaultdict(float)
    for path, weight in zip(paths, weights):
        for h, t, r in path:
            if (h,t,r) == (0,0,0):
                continue
            print(h, t, r, (h, t, r) in triplet2id)
            #print("not if",h, r, t)
            if r >= graph.num_relation:
                r = r - graph.num_relation.item()
                h, t = t, h
                print("in if", h, t, r)

            index = triplet2id[(h, t, r)]
            edge_weight[index] += weight
    edge_index, edge_weight = zip(*sorted(edge_weight.items()))

    graph = graph.edge_mask(edge_index)
    with graph.node():
        graph.original_node = torch.arange(graph.num_node)
    graph = graph.compact()
    graph._edge_weight = torch.tensor(edge_weight, device=graph.device)

    node_labels = []
    node_colors = {}
    h, t, r = sample[0].tolist()
    if r >= graph.num_relation:
        title = "p(%s | %s, %s^(-1))" % (entity_vocab[t], entity_vocab[h], relation_vocab[r - graph.num_relation.item()])
    else:
        title = "p(%s | %s, %s)" % (entity_vocab[t], entity_vocab[h], relation_vocab[r])
    if ranking is not None:
        title = "%s\nranking = %d" % (title, ranking)
    for i, index in enumerate(graph.original_node.tolist()):
        if index == h:
            node_colors[i] = "#ee6666"
        elif index == t:
            node_colors[i] = "#3ba272"
        node_labels.append(entity_vocab[index])

    plot.echarts(graph, title=title, node_colors=node_colors, node_labels=node_labels, relation_labels=relation_vocab,
                 dynamic_size=True, dynamic_width=True, save_file=save_file)


#util.setup_debug_hook()
torch.manual_seed(1024 + comm.get_rank())

logger = logging.getLogger(__name__)


vocab_file = os.path.join(os.path.dirname(__file__), "../data/PC_KEGG/PC_KEGG_CHEBI_entities.txt")
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

if __name__ == "__main__":
#    args = parse_args()
#    args.config = os.path.realpath(args.config)
#    cfgs = util.load_config(args.config, context=vars)
#
#    output_dir = util.create_working_directory(cfg)
#    if comm.get_rank() == 0:
#        logger = util.get_root_logger()    
#
#    logger.warning("Config file: %s" % args.config)
#
#    start = args.start or 0
#    end = args.end or len(cfg)
#    if comm.get_rank() == 0:
#        logger.warning("Config file: %s" % args.config)
#        logger.warning("Hyperparameter grid size: %d" % len(cfg))
#        logger.warning("Current job search range: [%d, %d)" % (start, end))
#        shutil.copyfile(args.config, os.path.basename(args.config))
#
#    cfg = cfg[start: end]
#    for job_id, cfg in enumerate(cfg):
#    for job_id, cfg in enumerate(cfg):
#    working_dir = output_dir
#    if len(cfg) > 1:
#        working_dir = os.path.join(working_dir, str(job_id))
#    if comm.get_rank() == 0:
#        #logger.warning("<<<<<<<<<< Job %d / %d start <<<<<<<<<<" % (job_id, len(cfg)))
#        logger.warning(pprint.pformat(cfg))
#        os.makedirs(working_dir, exist_ok=True)
#    comm.synchronize()
#    os.chdir(working_dir)

    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    _dataset = core.Configurable.load_config_dict(cfg.dataset)
    train_set, valid_set, test_set = _dataset.split()
    full_valid_set = valid_set
    if "NBFNet" in cfg.task.model["class"] and "CBKH" in cfg.dataset["class"]:
        valid_set = torch_data.random_split(valid_set, [500, len(valid_set) - 500])[0]
    if comm.get_rank() == 0:
        logger.warning(_dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    solver = build_solver(cfg)

    if "checkpoint" in cfg:
        solver.load(cfg.checkpoint)
#        state = torch.load(os.path.expanduser(cfg.checkpoint), map_location=solver.device)
#        state["model"].pop("fact_graph")
#        state["model"].pop("degree_hr")
#        state["model"].pop("degree_tr")
#        solver.model.load_state_dict(state["model"], strict=False)

    if "FB15k237" in cfg.dataset["class"]:
        entity_vocab, relation_vocab = 0(_dataset)
    else:
        entity_vocab, relation_vocab = load_vocab(_dataset)
#        entity_vocab = _dataset.entity_vocab
#        relation_vocab = _dataset.relation_vocab

    task = solver.model
    task.eval()
    for i in range(2):
        batch = data.graph_collate([test_set[i * solver.batch_size + j] for j in range(solver.batch_size)])
        batch = torchdrug.utils.cuda(batch)
        with torch.no_grad():
            pred, (mask, target) = task.predict_and_target(batch)
        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1

        for j in range(solver.batch_size):
            sample = batch[[j]]
            h, t, r = sample.flatten().tolist()
            


            entity = entity_vocab[h].replace(" ", "")
            relation = relation_vocab[r].replace(" ", "")
           
            
#            entity = re.search(r"(.+) \(Q\d+\)", entity_vocab[h]).groups()[0]
#            relation = re.search(r"(.+) \(\d+\)", relation_vocab[r]).groups()[0]
            save_file = "%s_%s.html" % (entity, relation)
            save_file = re.sub(r"[^\w().]+", "-", save_file)
            if ranking[j, 0] <= 10 and not os.path.exists(save_file):
                paths, weights = task.visualize(sample)
                if paths:
                    visualize_echarts(task.fact_graph, sample, paths, weights, entity_vocab, relation_vocab,
                                          ranking[j, 0], save_file)

#            entity = re.search(r"(.+) \(Q\d+\)", entity_vocab[t]).groups()[0]
            entity = entity_vocab[h].replace(" ", "")
            save_file = "%s_%s^(-1).html" % (entity, relation)
            save_file = re.sub(r"[^\w().]+", "-", save_file)
            sample = sample[:, [1, 0, 2]]
            sample[:, 2] += task.num_relation
            if ranking[j, 1] <= 10 and not os.path.exists(save_file):
                paths, weights = task.visualize(sample)
                if paths:
                        visualize_echarts(task.fact_graph, sample, paths, weights, entity_vocab, 
                                          relation_vocab, ranking[j, 1], save_file)
