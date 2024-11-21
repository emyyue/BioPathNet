import math

import torch
from torch.nn import functional as F
from torch.utils import data as torch_data

from ogb import linkproppred

from torchdrug import core, tasks, metrics
from torchdrug.layers import functional
from torchdrug.core import Registry as R
import torch_scatter
import numpy as np

from biopathnet import dataset


Evaluator = core.make_configurable(linkproppred.Evaluator)
Evaluator = R.register("ogb.linkproppred.Evaluator")(Evaluator)
setattr(linkproppred, "Evaluator", Evaluator)

  
@R.register("tasks.KnowledgeGraphCompletionBiomed")
class KnowledgeGraphCompletionBiomed(tasks.KnowledgeGraphCompletion, core.Configurable):

    def __init__(self, model, criterion="bce",
                 metric=("mr", "mrr", "hits@1", "hits@3", "hits@10", "hits@100", "hits@20","hits@50" , "auroc", "ap"),
                 num_negative=128, margin=6, adversarial_temperature=0, strict_negative=True,
                 heterogeneous_negative=False, heterogeneous_evaluation=False, filtered_ranking=True,
                 fact_ratio=None, sample_weight=True,
                 full_batch_eval=False, 
                 degree_negative = False, remove_pos=True,  train2_in_factgraph=True):
        super(KnowledgeGraphCompletionBiomed, self).__init__(model=model, criterion=criterion, metric=metric, 
                                                             num_negative=num_negative, margin=margin,
                                                             adversarial_temperature=adversarial_temperature, 
                                                             strict_negative=strict_negative,
                                                             filtered_ranking=filtered_ranking,fact_ratio=fact_ratio,
                                                             sample_weight=sample_weight, full_batch_eval=full_batch_eval)
        self.heterogeneous_negative = heterogeneous_negative
        self.heterogeneous_evaluation = heterogeneous_evaluation
        self.degree_negative = degree_negative
        self.remove_pos = remove_pos
        self.train2_in_factgraph = train2_in_factgraph
        
    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        self.register_buffer("graph", dataset.graph)
        fact_mask = torch.ones(len(dataset), dtype=torch.bool)
        fact_mask[valid_set.indices] = 0
        fact_mask[test_set.indices] = 0
        if self.fact_ratio:
            length = int(len(train_set) * self.fact_ratio)
            index = torch.randperm(len(train_set))[length:]
            train_indices = torch.tensor(train_set.indices)
            fact_mask[train_indices[index]] = 0
            train_set = torch_data.Subset(train_set, index)
            
        if self.train2_in_factgraph:
            self.register_buffer("fact_graph", dataset.graph.edge_mask(fact_mask))
            # get in degree per relation type
            self.in_degree_per_rel = self.get_in_degree_per_rel(
                self.fact_graph.undirected(add_inverse=True))
        else:
            # fact_graph_supervision is used to get negative samples - only remove valid and test
            self.register_buffer("fact_graph_supervision", dataset.graph.edge_mask(fact_mask))
            # get in degree per relation type
            ## for use in negative sampling
            self.in_degree_per_rel = self.get_in_degree_per_rel(
                self.fact_graph_supervision.undirected(add_inverse=True))
            
            # fact_graph is used for message passing
            fact_mask[train_set.indices] = 0
            self.register_buffer("fact_graph", dataset.graph.edge_mask(fact_mask))

        if self.sample_weight:
            degree_hr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            degree_tr = torch.zeros(self.num_entity, self.num_relation, dtype=torch.long)
            for h, t, r in train_set:
                degree_hr[h, r] += 1
                degree_tr[t, r] += 1
            self.register_buffer("degree_hr", degree_hr)
            self.register_buffer("degree_tr", degree_tr)
        return train_set, valid_set, test_set
        

    def get_in_degree_per_rel(self, graph):        
        ########################
        # making degree_in_type based on relations, as same nodes might have different relation types
        ########################

        # count the number of occurrence for each relation type for each node
        myindex = graph.edge_list[:, 0]
        relation_type = graph.edge_list[:, 2]
        # one hot encoding of relation types
        # Zhaocheng: myinput is (|E|, |R|), potentially OOM for large graphs
        # You may augment myindex to be relation_type * graph.num_node + myindex
        # and scatter_add ones with the augmented index
        # finally reshape the tensor from (num_relation * num_node,) to (num_relation, num_node)
        myinput = torch.t(F.one_hot(relation_type))
        # calculate
        degree_in_type = myinput.new_zeros(graph.num_relation,  graph.num_node) # which output dim
        degree_in_type = torch_scatter.scatter_add(myinput, myindex, out=degree_in_type)
        
        return degree_in_type
        
        
    def target(self, batch):
        # test target
        batch_size = len(batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        any = -torch.ones_like(pos_h_index)
        node_type = self.fact_graph.node_type

        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        edge_index, num_t_truth = self.graph.match(pattern)
        t_truth_index = self.graph.edge_list[edge_index, 1]
        pos_index = torch.repeat_interleave(num_t_truth)
        t_mask = torch.ones(batch_size, self.num_entity, dtype=torch.bool, device=self.device)
        if self.remove_pos:
            t_mask[pos_index, t_truth_index] = 0
        if self.heterogeneous_evaluation:
            t_mask[node_type.unsqueeze(0) != node_type[pos_t_index].unsqueeze(-1)] = 0

        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        edge_index, num_h_truth = self.graph.match(pattern)
        h_truth_index = self.graph.edge_list[edge_index, 0]
        pos_index = torch.repeat_interleave(num_h_truth)
        h_mask = torch.ones(batch_size, self.num_entity, dtype=torch.bool, device=self.device)
        if self.remove_pos:
            h_mask[pos_index, h_truth_index] = 0
        if self.heterogeneous_evaluation:
            h_mask[node_type.unsqueeze(0) != node_type[pos_h_index].unsqueeze(-1)] = 0

        mask = torch.stack([t_mask, h_mask], dim=1)
        target = torch.stack([pos_t_index, pos_h_index], dim=1)

        # in case of GPU OOM
        return mask.cpu(), target.cpu()

    def evaluate(self, pred, target):
        mask, target = target

        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        if self.filtered_ranking:
            ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        else:
            ranking = torch.sum(pos_pred <= pred, dim=-1) + 1 
        # get neg predictions
        # remove the true tail and head
        m = torch.ones_like(pred).scatter(2, target.unsqueeze(-1), 0).bool()
        # use mask to get rid of other true tails and heads
        prob =  m.logical_and(~mask).long().float()
        # sample from neg exampels
        neg_t = functional.multinomial(prob[:,0,:], 1, replacement=True)
        neg_h = functional.multinomial(prob[:,1,:], 1, replacement=True)
        # concat and get mask
        mask_neg =  torch.cat((neg_t, neg_h), -1)
        # get neg predictions
        neg_pred = pred.gather(-1, mask_neg.unsqueeze(-1))
        # take random sample of neg predictions
        pred = torch.stack((pos_pred.flatten(), neg_pred.flatten()),1)
        # construct the target out of positive (1) and negative (0)
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        pred = pred.flatten()
        target = target.flatten()
        
        metric = {}
        
        for _metric in self.metric:
            if _metric == "mr":
                score = ranking.float().mean()
            elif _metric == "mrr":
                score = (1 / ranking.float()).mean()
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                score = (ranking <= threshold).float().mean()
            elif _metric == "auroc":
                score = metrics.area_under_roc(pred, target)
            elif _metric == "ap":
                score = metrics.area_under_prc(pred, target)
            else:
                continue
                #raise ValueError("Unknown metric `%s`" % _metric)
            name = tasks._get_metric_name(_metric)
            metric[name] = score
            
        return metric
    
    def predict(self, batch, all_loss=None, metric=None):
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        batch_size = len(batch)

        if all_loss is None:
            # test
            all_index = torch.arange(self.num_entity, device=self.device) # evaluate against all nodes
            t_preds = []
            h_preds = []
            num_negative = self.num_entity if self.full_batch_eval else self.num_negative
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                h_index, t_index = torch.meshgrid(pos_h_index, neg_index)
                t_pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
                t_preds.append(t_pred)
            t_pred = torch.cat(t_preds, dim=-1)
            for neg_index in all_index.split(num_negative):
                r_index = pos_r_index.unsqueeze(-1).expand(-1, len(neg_index))
                t_index, h_index = torch.meshgrid(pos_t_index, neg_index)
                h_pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)

                h_preds.append(h_pred)
                
            h_pred = torch.cat(h_preds, dim=-1)
            pred = torch.stack([t_pred, h_pred], dim=1)
            # in case of GPU OOM
            pred = pred.cpu()
                
        else:
            # train
            if self.strict_negative:
                neg_index = self._strict_negative(pos_h_index, pos_t_index, pos_r_index)
            else:
                neg_index = torch.randint(self.num_entity, (batch_size, self.num_negative), device=self.device)
            h_index = pos_h_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(1, self.num_negative + 1)
            t_index[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
            h_index[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
            pred = self.model(self.fact_graph, h_index, t_index, r_index, all_loss=all_loss, metric=metric)
        return pred
    

    
    @torch.no_grad()
    def _strict_negative(self, pos_h_index, pos_t_index, pos_r_index, degree_in_type=None, num_nodes_per_type=None, graph=None):
        batch_size = len(pos_h_index)
        any = -torch.ones_like(pos_h_index)
        node_type = self.fact_graph.node_type
        degree_in_rel = self.in_degree_per_rel
        
        ####################### 
        # sample negative heads # (pos_h, r, neg_t)
        ####################### 
        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        pattern = pattern[:batch_size // 2]
        # if train2 not used for mp, it should still serve to sample negatives
        if self.train2_in_factgraph:
            edge_index, num_t_truth = self.fact_graph.match(pattern)
            t_truth_index = self.fact_graph.edge_list[edge_index, 1]
        else:
            edge_index, num_t_truth = self.fact_graph_supervision.match(pattern)
            t_truth_index = self.fact_graph_supervision.edge_list[edge_index, 1]
        pos_index = torch.repeat_interleave(num_t_truth)

        # remove undesired node type
        if self.heterogeneous_negative:
            pos_t_type = node_type[pos_t_index[:batch_size // 2]]
            t_mask = pos_t_type.unsqueeze(-1) == node_type.unsqueeze(0)
        else:
            t_mask = torch.ones(len(pattern), self.num_entity, dtype=torch.bool, device=self.device)
        # remove positive
        t_mask[pos_index, t_truth_index] = 0
        
        ## sample tails not in fact graph
        if self.degree_negative:
            # multinomial negative sampling according to degree per rel
            # get degree per relation type
            degree_in_rel_expanded = degree_in_rel[pattern[:,2] + self.graph.num_relation].float()
            prob_deg = torch.zeros_like(degree_in_rel_expanded, dtype=torch.float)  # Initialize a result tensor
            # apply mask
            prob_deg[t_mask] = (degree_in_rel_expanded[t_mask] + 1)
            neg_t_index = functional.multinomial(prob_deg, self.num_negative, replacement=True)
        else:
            # variadic sampling of negatives uniformly
            # get the candidates in one list
            neg_t_candidate = t_mask.nonzero()[:, 1]
            # get number of candidates belonging to each triplet
            num_t_candidate = t_mask.sum(dim=-1)
            # sample num_negative from candidate list, knowing (over neg_t_candidate) how many belong to each triplet
            neg_t_index = functional.variadic_sample(neg_t_candidate, num_t_candidate, self.num_negative)
            


        ####################### 
        # sample negative heads # (neg_h, r-1, pos_t)
        ####################### 
        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        pattern = pattern[batch_size // 2:]
        # if train2 is used for mp
        if self.train2_in_factgraph:
            edge_index, num_h_truth = self.fact_graph.match(pattern)
            h_truth_index = self.fact_graph.edge_list[edge_index, 0]
        else:
            edge_index, num_h_truth = self.fact_graph_supervision.match(pattern)
            h_truth_index = self.fact_graph_supervision.edge_list[edge_index, 0]
            
        pos_index = torch.repeat_interleave(num_h_truth)
        
        # remove undesired node type
        if self.heterogeneous_negative:
            pos_h_type = node_type[pos_h_index[batch_size // 2:]]
            h_mask = pos_h_type.unsqueeze(-1) == node_type.unsqueeze(0)
        else:
            h_mask = torch.ones(len(pattern), self.num_entity, dtype=torch.bool, device=self.device)
        # remove positive
        h_mask[pos_index, h_truth_index] = 0
        
        if self.degree_negative:
            # multinomial negative sampling according to degree per rel
            degree_in_rel_expanded = degree_in_rel[pattern[:,2]].float()
            prob_deg = torch.zeros_like(degree_in_rel_expanded, dtype=torch.float)  # Initialize a result tensor
            # apply mask
            prob_deg[h_mask] = (degree_in_rel_expanded[h_mask] + 1)
            neg_h_index = functional.multinomial(prob_deg, self.num_negative, replacement=True)
        else:
            # variadic sampling of negatives uniformly
            neg_h_candidate = h_mask.nonzero()[:, 1]
            num_h_candidate = h_mask.sum(dim=-1)
            neg_h_index = functional.variadic_sample(neg_h_candidate, num_h_candidate, self.num_negative)
        
        neg_index = torch.cat([neg_t_index, neg_h_index])
        return neg_index
    
    
@R.register("tasks.KnowledgeGraphCompletionBiomedEval")
class KnowledgeGraphCompletionBiomedEval(KnowledgeGraphCompletionBiomed, core.Configurable):
    def __init__(self, model, criterion="bce",
                metric=("mr", "mrr", "hits@1", "hits@3", "hits@10", "hits@100", "auroc", "ap", "auroc_all", "ap_all"),
                num_negative=128, margin=6, adversarial_temperature=0, strict_negative=True,
                heterogeneous_negative=False, heterogeneous_evaluation=False, filtered_ranking=True,
                fact_ratio=None, sample_weight=True,
                full_batch_eval=False):
        super(KnowledgeGraphCompletionBiomedEval, self).__init__(model=model, criterion=criterion, metric=metric, 
                                                                num_negative=num_negative, margin=margin,
                                                                adversarial_temperature=adversarial_temperature, 
                                                                strict_negative=strict_negative,
                                                                heterogeneous_negative=heterogeneous_negative,
                                                                heterogeneous_evaluation=heterogeneous_evaluation,
                                                                filtered_ranking=filtered_ranking,fact_ratio=fact_ratio,
                                                                sample_weight=sample_weight, full_batch_eval=full_batch_eval)
    
    def evaluate(self, pred, target): 
        mask, target = target

        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        # get ranking per source node

        ranking_filt = ranking.new_zeros(mask.shape[1], mask.shape[2]).float()
        ranking_filt = torch_scatter.scatter_mean(torch.transpose(ranking, 0, 1).float(),
                                                  torch.flip(torch.transpose(target, 0, 1), [0]), out=ranking_filt)
        valid_ranking =  np.ma.masked_where(ranking_filt == 0, ranking_filt)
        print(f'Rankings aggregated per tail node {valid_ranking[1][~valid_ranking[1].mask]}')
        print(f'MRR per tail node {1/(valid_ranking[1][~valid_ranking[1].mask])}')
        print(f'index of tail nodes {[i for i, x in enumerate(~valid_ranking[1].mask) if x]}')

        # get neg_pred
        mask_inv_target = torch.ones_like(pred, dtype=torch.bool)
        mask_inv_target.scatter_(-1, target.unsqueeze(-1), False) # filtered setting: add testing mask
        mask_inv_target = mask_inv_target & mask # add mask from previous
        
        
        # nodes t
        trans_target = torch.transpose(torch.flip(torch.transpose(target, 0, 1), [0]), 0,1)
        nodes_t = trans_target[:,0].unique()
        pred_t_auprc = []
        pred_t_auroc = []
        for i in nodes_t:
            # pos pred per h node
            idx1 = (trans_target[:,0] == i).nonzero().squeeze(-1)
            idx3 = target[idx1][:,0]
            pos_pred_node = pred[idx1, 0, idx3]
            # neg pred per h node
            neg_pred_node = pred[idx1[0], 0, :].masked_select(mask_inv_target[idx1[0], 0,:])
            # assemble
            pred_node = torch.concat([pos_pred_node, neg_pred_node])
            gt = torch.cat([torch.ones_like(pos_pred_node),torch.zeros_like(neg_pred_node)])
            # calculate
            pred_t_auprc.append(metrics.area_under_prc(pred_node, gt))
            pred_t_auroc.append(metrics.area_under_roc(pred_node, gt))

        pred_t_auprc_mean = np.array(pred_t_auprc).mean()
        pred_t_auroc_mean = np.array(pred_t_auroc).mean()
        
        # nodes h
        nodes_h = trans_target[:,1].unique()
        pred_h_auprc = []
        pred_h_auroc = []
        for i in nodes_h:
            idx1 = (trans_target[:,1] == i).nonzero().squeeze(-1)
            idx3 = target[idx1][:,1] 
            pos_pred_node = pred[idx1, 1, idx3]
            neg_pred_node = pred[idx1[0], 1, :].masked_select(mask_inv_target[idx1[0], 1,:])
            pred_node = torch.concat([pos_pred_node, neg_pred_node])
            gt = torch.cat([torch.ones_like(pos_pred_node),torch.zeros_like(neg_pred_node)])
            pred_h_auprc.append(metrics.area_under_prc(pred_node, gt))
            pred_h_auroc.append(metrics.area_under_roc(pred_node, gt))

        pred_h_auprc_mean = np.array(pred_h_auprc).mean()
        pred_h_auroc_mean = np.array(pred_h_auroc).mean()
        
        # calculate auroc and auprc for all predictions (instead of 1:1)
        # split into t and h neg_pred
        neg_pred_t = pred[:,0,:].masked_select(mask_inv_target[:,0,:]) 
        neg_pred_h = pred[:,1,:].masked_select(mask_inv_target[:,1,:]) 
        # get for t and h the predictions
        pred_metric_t = torch.cat([pos_pred[:,0,:].flatten(), neg_pred_t.flatten()])
        pred_metric_h = torch.cat([pos_pred[:,1,:].flatten(), neg_pred_h.flatten()])
        # construct for t and h, the pos and neg labels
        target_metric_t = torch.cat([torch.ones_like(pos_pred[:,0,:].flatten()),torch.zeros_like(neg_pred_t)])
        target_metric_h = torch.cat([torch.ones_like(pos_pred[:,1,:].flatten()),torch.zeros_like(neg_pred_h)])
        
        
        
        metric = {}
        for _metric in self.metric:
            if _metric == "mr":
                score = valid_ranking.mean(1).data
            elif _metric == "mrr":
                score = (1/valid_ranking).mean(1).data
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                score = ((1 - np.ma.masked_where(valid_ranking >= threshold,
                                        valid_ranking).mask).sum(1))/((1- valid_ranking.mask).sum(1))
            elif _metric == "auroc":
                score = np.array([pred_t_auroc_mean, pred_h_auroc_mean])
            elif _metric == "ap":
                score = np.array([pred_t_auprc_mean, pred_h_auprc_mean])
            elif _metric == "auroc_all":
                score = np.array([metrics.area_under_roc(pred_metric_t, target_metric_t),
                                  metrics.area_under_roc(pred_metric_h, target_metric_h)])
            elif _metric == "ap_all":
                score = np.array([metrics.area_under_prc(pred_metric_t, target_metric_t) ,
                                  metrics.area_under_prc(pred_metric_h, target_metric_h) ])
            else:
                raise ValueError("Unknown metric `%s`" % _metric)
            name = tasks._get_metric_name(_metric)
            name_t = name + '_t'
            name_h = name + '_h'
            metric[name_t] = score[0]
            metric[name_h] = score[1]
            
        return metric
    
    
    
    
    def target(self, batch):
        ####
        # for evaluation of TxGNN, heterogeneous evaluation should be switched on
        # filtered ranking is not important to be on yes, 
        # as there should be no drugs in training data at all
        ####
        # test target
        batch_size = len(batch)
        pos_h_index, pos_t_index, pos_r_index = batch.t()
        any = -torch.ones_like(pos_h_index)
        node_type = self.fact_graph.node_type

        t_mask = torch.ones(batch_size, self.num_entity, dtype=torch.bool, device=self.device)
        if self.filtered_ranking:
            pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
            edge_index, num_t_truth = self.graph.match(pattern)
            t_truth_index = self.graph.edge_list[edge_index, 1]
            pos_index = torch.repeat_interleave(num_t_truth)
            t_mask[pos_index, t_truth_index] = 0
        if self.heterogeneous_evaluation:
            t_mask[node_type.unsqueeze(0) != node_type[pos_t_index].unsqueeze(-1)] = 0

        h_mask = torch.ones(batch_size, self.num_entity, dtype=torch.bool, device=self.device)
        if self.filtered_ranking:
            pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
            edge_index, num_h_truth = self.graph.match(pattern)
            h_truth_index = self.graph.edge_list[edge_index, 0]
            pos_index = torch.repeat_interleave(num_h_truth)
            h_mask[pos_index, h_truth_index] = 0
        if self.heterogeneous_evaluation:
            h_mask[node_type.unsqueeze(0) != node_type[pos_h_index].unsqueeze(-1)] = 0

        mask = torch.stack([t_mask, h_mask], dim=1)
        target = torch.stack([pos_t_index, pos_h_index], dim=1)

        # in case of GPU OOM
        return mask.cpu(), target.cpu()
