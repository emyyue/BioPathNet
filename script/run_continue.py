import os
import sys
import pprint

import torch
import math
import random


from torchdrug import core
from torchdrug.utils import comm
from torch.utils import data as torch_data
from torchdrug import data, core, utils

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from biopathnet import dataset, layer, model, task, util#, reasoning_mod
import numpy as np


def solver_load(checkpoint, load_optimizer=True):

    if comm.get_rank() == 0:
        logger.warning("Load checkpoint from %s" % checkpoint)
    checkpoint = os.path.expanduser(checkpoint)
    state = torch.load(checkpoint, map_location=solver.device)
    # some issues with loading back the fact_graph and graph
    # remove
    state["model"].pop("fact_graph", 0)
    state["model"].pop("fact_graph_supervision", 0)
    state["model"].pop("graph", 0)
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
    cfg.task.model.num_relation = dataset.num_relation
    _task = core.Configurable.load_config_dict(cfg.task)
    cfg.optimizer.params = _task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    if "scheduler" in cfg:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
    else:
        scheduler = None
    return core.Engine(_task, train_set, valid_set, test_set, optimizer, scheduler, **cfg.engine)

def load_vocab(dataset, vocab_file):
    entity_mapping = {}
    with open(vocab_file, "r") as fin:
        for line in fin:
            k, v = line.strip().split("\t")
            entity_mapping[k] = v
    entity_vocab = [entity_mapping[t] for t in dataset.entity_vocab]
    relation_vocab = ["%s (%d)" % (t[t.rfind("/") + 1:].replace("_", " "), i)
                      for i, t in enumerate(dataset.relation_vocab)]

    return entity_vocab, relation_vocab

def train_and_validate(cfg, solver):
    if cfg.train.num_epoch == 0:
        return

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.model.split = "train"
        solver.train(**kwargs)
        solver.save("model_epoch_%d.pth" % solver.epoch)
        solver.model.split = "valid"
        metric = solver.evaluate("valid")
        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch

    solver_load("model_epoch_%d.pth" % best_epoch)
    return solver


def test(cfg, solver):
    solver.model.split = "valid"
    solver.evaluate("valid")
    solver.model.split = "test"
    solver.evaluate("test")

        
if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)
    args.seed = int(args.seed)
    seed_rank = args.seed + int(comm.get_rank())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(seed_rank)

    logger = util.get_root_logger()
    logger.warning("Working directory: %s" % working_dir)
    logger.warning("Input Seed: %d" % args.seed)
    logger.warning("Set Seed: %d" % seed_rank)
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    train_set, valid_set, test_set = dataset.split()
    solver = build_solver(cfg)

    if "checkpoint" in cfg:
        solver_load(cfg.checkpoint)
    logger.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    logger.warning("Loaded checkpoint and evaluate on valid and test")
    test(cfg, solver)
    logger.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    logger.warning("Loaded checkpoint and continue training")
    logger.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    train_and_validate(cfg, solver)
    test(cfg, solver)
