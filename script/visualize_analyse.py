import os
import sys
import pprint
import pickle
import pandas as pd

import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from biopathnet import dataset, layer, model, task, util




def solver_load(checkpoint, load_optimizer=True):

    if comm.get_rank() == 0:
        logger.warning("Load checkpoint from %s" % checkpoint)
    checkpoint = os.path.expanduser(checkpoint)
    state = torch.load(checkpoint, map_location=solver.device)
    # some issues with loading back the graphs if present
    # remove
    state["model"].pop("fact_graph", 0)
    state["model"].pop("fact_graph_supervision", 0)
    state["model"].pop("graph", 0)
    state["model"].pop("train_graph", 0)
    state["model"].pop("valid_graph", 0)
    state["model"].pop("test_graph", 0)
    state["model"].pop("full_valid_graph", 0)
    state["model"].pop("full_test_graph", 0)
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

def visualize_path(solver, triplet, entity_vocab, relation_vocab):
    num_relation = len(relation_vocab)
    h, t, r = triplet.tolist()
    triplet = torch.as_tensor([[h, t, r]], device=solver.device)
    inverse = torch.as_tensor([[t, h, r + num_relation]], device=solver.device)
    solver.model.eval()
    pred, (mask, target) = solver.model.predict_and_target(triplet)
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    rankings = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    rankings = rankings.squeeze(0)

    logger.warning("")
    samples = (triplet, inverse)
    all_paths = []
    all_weights = []
    for sample, ranking in zip(samples, rankings):
        h, t, r = sample.squeeze(0).tolist()
        h_name = entity_vocab[h]
        t_name = entity_vocab[t]
        r_name = relation_vocab[r % num_relation]
        if r >= num_relation:
            r_name += "^(-1)"
        logger.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.warning("rank(%s | %s, %s) = %g" % (t_name, h_name, r_name, ranking))

        paths, weights = solver.model.visualize(sample)
        for path, weight in zip(paths, weights):
            triplets = []
            for h, t, r in path:
                h_name = entity_vocab[h]
                t_name = entity_vocab[t]
                r_name = relation_vocab[r % num_relation]
                if r >= num_relation:
                    r_name += "^(-1)"
                triplets.append("<%s, %s, %s>" % (h_name, r_name, t_name))
            logger.warning("weight: %g\n\t%s" % (weight, " ->\n\t".join(triplets)))
        all_paths.append(paths)
        all_weights.append(weights)
    return all_paths, all_weights

def paths_to_table(results):
    rows = []
    for sample_id, (sample_paths, sample_weights) in enumerate(zip(results['paths'], results['weights'])):
        forward_paths, reverse_paths = sample_paths
        forward_weights, reverse_weights = sample_weights

        # Forward
        for path_id, (path, weight) in enumerate(zip(forward_paths, forward_weights)):
            for step_id, (h, t, r) in enumerate(path):
                rows.append((sample_id, "forward", path_id, step_id, h, r, t, weight))
        
        # Reverse
        for path_id, (path, weight) in enumerate(zip(reverse_paths, reverse_weights)):
            for step_id, (h, t, r) in enumerate(path):
                rows.append((sample_id, "reverse", path_id, step_id, h, r, t, weight))

    df = pd.DataFrame(rows, columns=[
        "sample_id", "direction", "path_id", "step_id", "h", "r", "t", "edge_weight"
    ])
    df["index"] = df.index
    return df


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)
    print(working_dir)
    vocab_file = os.path.join(os.path.dirname(__file__), cfg.dataset.path, "entity_names.txt")
    vocab_file = os.path.abspath(vocab_file)
    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    logger.warning("Working directory: %s" % working_dir)
    logger.warning("Config file: %s" % args.config)
    logger.warning(pprint.pformat(cfg))

    cfg.dataset.files = ['train1.txt', 'train2.txt', 'valid.txt', 'test_vis.txt']
    _dataset = core.Configurable.load_config_dict(cfg.dataset)
    train_set, valid_set, test_set = _dataset.split()
    solver = build_solver(cfg)
    
    if "checkpoint" in cfg:
        solver_load(cfg.checkpoint)
        
    entity_vocab, relation_vocab = load_vocab(_dataset)

    all_paths = []
    all_weights = []
    for i in range(len(solver.test_set)):
        paths, weights = visualize_path(solver, solver.test_set[i], entity_vocab, relation_vocab)
        all_paths.append(paths)
        all_weights.append(weights)
    results = {
        'weights': all_weights,
        'paths': all_paths,
        'entity_vocab': _dataset.entity_vocab,
        'entity_vocab_names': entity_vocab,
        'relation_vocab': _dataset.relation_vocab
        }

    with open(os.path.join( working_dir, "visualize_analyse.pkl"), "wb") as f:
        pickle.dump(results, f)
