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
        graph.original_node = torch.arange(graph.num_node, device=graph.device)
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
        
    triplet2id = {tuple(edge.tolist()): i for i, edge in enumerate(graph.edge_list)}

    edge_colors = {}
    for h, t, r in paths[0]:
        h = graph.original_node.tolist().index(h)
        t = graph.original_node.tolist().index(t)
        if (h,t,r) == (0,0,0):
            continue
        if r >= graph.num_relation:
            r = r - graph.num_relation.item()
            h, t = t, h
            print("in if", h, t, r)
        index = triplet2id[(h, t, r)]
        edge_colors[index] = "#F14167"
    
    node_type = graph.node_type
    node_colors_dict = {0: "#72568f",
                        1: "#f9844a",
                        2: "#577590",
                        3: "#277da1",
                        4: "#f3722c",
                        5: "#f94144",
                        6: "#43aa8b",
                        7: "#f9c74f",
                        8: "#f8961e",
                        9: "#f94144"}
    for i, index in enumerate(graph.original_node.tolist()):
        node_colors[i] = node_colors_dict[node_type[i].cpu().item()]
        node_labels.append(entity_vocab[index])
    
    h, t, r = sample[0].tolist()    
    for i, index in enumerate(graph.original_node.tolist()):
        if index == h:
            node_colors[i] = "#F9B895"
        elif index == t:
            node_colors[i] = "#82C4E1"

    plot.echarts(graph, title=title, node_colors=node_colors, node_labels=node_labels, relation_labels=relation_vocab,
                 edge_colors=edge_colors,
                 dynamic_size=True, dynamic_width=True, save_file=save_file)


#util.setup_debug_hook()
torch.manual_seed(1024 + comm.get_rank())

logger = logging.getLogger(__name__)


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
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)
    print(working_dir)
    vocab_file = os.path.join(os.path.dirname(__file__), cfg.dataset.path, "entity_names.txt")
    vocab_file = os.path.abspath(vocab_file)
    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    cfg.dataset.files = ['train1.txt', 'train2.txt', 'valid.txt', 'test_eval.txt']
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
        solver_load(cfg.checkpoint)


    if "FB15k237" in cfg.dataset["class"]:
        entity_vocab, relation_vocab = 0(_dataset)
    else:
        entity_vocab, relation_vocab = load_vocab(_dataset)


    task = solver.model
    task.eval()
    for i in range(20):
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
            entity_t = entity_vocab[t].replace(" ", "")
           
            
#            entity = re.search(r"(.+) \(Q\d+\)", entity_vocab[h]).groups()[0]
#            relation = re.search(r"(.+) \(\d+\)", relation_vocab[r]).groups()[0]
            save_file = "%s_%s_%s.html" % (entity, relation, entity_t)
            save_file = re.sub(r"[^\w().]+", "-", save_file)
            if ranking[j, 0] <= 10 and not os.path.exists(save_file):
                paths, weights = task.visualize(sample)
                if paths:
                    visualize_echarts(task.fact_graph, sample, paths, weights, entity_vocab, relation_vocab,
                                          ranking[j, 0], save_file)

#            entity = re.search(r"(.+) \(Q\d+\)", entity_vocab[t]).groups()[0]
            entity = entity_vocab[h].replace(" ", "")
            save_file = "%s_%s^(-1)_%s.html" % (entity_t, relation, entity)
            save_file = re.sub(r"[^\w().]+", "-", save_file)
            sample = sample[:, [1, 0, 2]]
            sample[:, 2] += task.num_relation
            if ranking[j, 1] <= 10 and not os.path.exists(save_file):
                paths, weights = task.visualize(sample)
                if paths:
                    visualize_echarts(task.fact_graph, sample, paths, weights, entity_vocab, 
                                          relation_vocab, ranking[j, 1], save_file)

