import os
import sys
import pprint

import torch

from torchdrug import core
from torchdrug.utils import comm
from torch.utils import data as torch_data
from torchdrug import data, core, utils

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model, task, util
import numpy as np


@torch.no_grad()
def get_prediction(cfg, solver):
    test_set = solver.test_set

    dataloader = data.DataLoader(test_set, solver.batch_size, sampler=None, num_workers=solver.num_worker)
    model = solver.model

    model.eval()
    preds = []
    targets = []
    
    print("before loop")
    for ith, batch in enumerate(dataloader):

        if solver.device.type == "cuda":
            batch = utils.cuda(batch, device=solver.device)
            
        print(f"We are at batch {ith}")
        #import pdb; pdb.set_trace()
        pred, target = model.predict_and_target(batch)
        print(f"I got the predictions for batch {ith}")
        
        preds.append(pred)
        targets.append(target)
        print("now preds")
        print(preds)
        print("now targets")
        print(targets)


    
    pred = utils.cat(preds)
    target = utils.cat(targets)
    print("now pred")
    print(pred)


    print("now target")
    print(target)
    
    return pred, target

        
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

    relation_vocab = ["%s (%d)" % (t[t.rfind("/") + 1:].replace("_", " "), i)
                      for i, t in enumerate(dataset.relation_vocab)]
    #import ipdb; ipdb.set_trace()
    pred, target = get_prediction(cfg, solver)
    
    #save
    arr = pred.data.cpu().numpy()
    # write CSV
    np.savetxt('output.csv', arr)