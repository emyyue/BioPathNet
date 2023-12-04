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
import pandas as pd



import os
import json
import jinja2
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

path = os.path.join(os.path.dirname(__file__), "template")
def echarts(graph, title=None, node_colors=None, edge_colors=None, node_labels=None, relation_labels=None,
            node_types=None, type_labels=None, dynamic_size=False, dynamic_width=False, save_file=None):
    """
    Visualize a graph in ECharts.

    Parameters:
        graph (Graph): graph to visualize
        title (str, optional): title of the graph
        node_colors (dict, optional): specify colors for some nodes.
            Each color is either a tuple of 3 integers between 0 and 255, or a hex color code.
        edge_colors (dict, optional): specify colors for some edges.
            Each color is either a tuple of 3 integers between 0 and 255, or a hex color code.
        node_labels (list of str, optional): labels for each node
        relation_labels (list of str, optional): labels for each relation
        node_types (list of int, optional): type for each node
        type_labels (list of str, optional): labels for each node type
        dynamic_size (bool, optional): if true, set the size of nodes based on the logarithm of degrees
        dynamic_width (bool, optional): if true, set the width of edges based on the edge weights
        save_file (str, optional): ``html`` file to save visualization, accompanied by a ``json`` file
    """
    if dynamic_size:
        symbol_size = (graph.degree_in + graph.degree_out + 2).log()
        symbol_size = symbol_size / symbol_size.mean() * 10
        symbol_size = symbol_size.tolist()
    else:
        symbol_size = [10] * graph.num_node
    nodes = []
    node_colors = node_colors or {}
    for i in range(graph.num_node):
        node = {
            "id": i,
            "symbolSize": symbol_size[i],
        }
        if i in node_colors:
            color = node_colors[i]
            if isinstance(color, tuple):
                color = "rgb%s" % (color,)
            node["itemStyle"] = {"color": color}
        if node_labels:
            node["name"] = node_labels[i]
        if node_types:
            node["category"] = node_types[i]
        nodes.append(node)

    if dynamic_width:
        width = graph.edge_weight / graph.edge_weight.mean() * 3
        width = width.tolist()
    else:
        width = [3] * graph.num_edge
    edges = []
    if graph.num_relation:
        node_in, node_out, relation = graph.edge_list.t().tolist()
    else:
        node_in, node_out = graph.edge_list.t().tolist()
        relation = None
    edge_colors = edge_colors or {}
    for i in range(graph.num_edge):
        edge = {
            "source": node_in[i],
            "target": node_out[i],
            "lineStyle": {"width": width[i]},
        }
        if i in edge_colors:
            color = edge_colors[i]
            if isinstance(color, tuple):
                color = "rgb%s" % (color,)
            edge["lineStyle"].update({"color": color})
        if relation_labels:
            edge["value"] = relation_labels[relation[i]]
        edges.append(edge)

    json_file = os.path.splitext(save_file)[0] + ".json"
    data = {
        "title": title,
        "nodes": nodes,
        "edges": edges,
    }
    if type_labels:
        data["categories"] = [{"name": label} for label in type_labels]
    variables = {
        "data_file": os.path.basename(json_file),
        "show_label": "true" if node_labels else "false",
    }
    with open(os.path.join(path, "echarts.html"), "r") as fin, open(save_file, "w") as fout:
        template = jinja2.Template(fin.read())
        instance = template.render(variables)
        fout.write(instance)
    with open(json_file, "w") as fout:
        json.dump(data, fout, sort_keys=True, indent=4)

def visualize_echarts(graph, sample, paths, weights, entity_vocab, relation_vocab, ranking=None, save_file=None, node_colors_dict=None):
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
        edge_colors[index] = node_colors_dict[99]
    
    node_type = graph.node_type
    if node_colors_dict:        
        for i, index in enumerate(graph.original_node.tolist()):
            node_colors[i] = node_colors_dict[node_type[i].cpu().item()]
            node_labels.append(entity_vocab[index])

    
    # different color for head and tail 
    # h, t, r = sample[0].tolist()    
    # for i, index in enumerate(graph.original_node.tolist()):
    #     if index == h:
    #         node_colors[i] = "#F9B895"
    #     elif index == t:
    #         node_colors[i] = "#82C4E1"
            
    # add edge, edge weight and edge color
    # h = graph.original_node.tolist().index(h)
    # t = graph.original_node.tolist().index(t)
    # if r >= graph.num_relation:
    #     r = r - graph.num_relation.item()
    #     h, t = t, h
        

    # with graph.edge():
    #     graph.edge_list = torch.tensor(graph.edge_list.tolist() + [[h,t,r]],
    #                                    device=graph.device)
    #     graph._edge_weight = torch.tensor(edge_weight + tuple([max(edge_weight)]),
    #                                       device=graph.device)
    
    echarts(graph, title=title, node_colors=node_colors, node_labels=node_labels, relation_labels=relation_vocab,
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
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), cfg.dataset.path, "node_colors_dict.txt"), sep="\t")
    node_colors_dict = df.set_index('type').T.to_dict(orient="index")['color']

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
                                          ranking[j, 0], save_file, node_colors_dict=node_colors_dict)

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
                                          relation_vocab, ranking[j, 1], save_file, node_colors_dict=node_colors_dict)

