import os
import pandas as pd
import numpy as np
import ipdb

from scipy.stats import rankdata
import subprocess
import logging


def cal_ndcg(scores, labels, rm_list, n=10):
    # ndcg topk
    denom = np.log2(np.arange(2, n + 2))
    ndcgs = []
    mrrs = []
    p_topks = []
    r_topks = []
    hit_topks = []
    hit_at_ks = []
    for i in range(len(scores)):
        scores_i = scores[i]
        sorted_list = np.argsort(scores_i, axis=0, kind='stable')[::-1]
        
        if rm_list[i] == 'set()':
            rm_list_tmp = list(eval(rm_list[i]))
        else: 
            rm_list_tmp = np.array(rm_list[i].strip('}{').split(', '), dtype=np.int64)

        sorted_list_tmp = [s for s in sorted_list if s not in rm_list_tmp]
        gt = np.nonzero(labels[i])[0]
        
        # for MRR:
        labels_i = labels[i]
        index_list = list(range(len(scores_i)))
        filtered_list_tmp = [s for s in index_list if s not in rm_list_tmp]
        filtered_labels_i = labels_i[filtered_list_tmp]
        filtered_scores_i = scores_i[filtered_list_tmp]
        pred_pos = filtered_scores_i[np.where(filtered_labels_i == 1)[0]]
        filtered_scores_i = np.delete(filtered_scores_i, np.where(filtered_labels_i == 1)[0])
        n_neg = []
        for p in range(len(pred_pos)):
            n_neg.append(sum(filtered_scores_i >= pred_pos[p]) + 1)
        mrr = (1/np.array(n_neg)).mean()
       

        hit_topk = len(np.intersect1d(sorted_list_tmp[:n], gt))
        hit_at_k = hit_topk/n

        dcg_topk = np.sum(np.in1d(sorted_list_tmp[:n], gt) / denom)
        idcg_topk = np.sum((1 / denom)[:min(len(gt), n)])
        ndcg = dcg_topk / idcg_topk if idcg_topk != 0 else 0
        p_topk = hit_topk / min(len(gt), n) if len(gt) != 0 else 0
        r_topk = hit_topk / len(gt) if len(gt) != 0 else 0
        ndcgs.append(ndcg)
        p_topks.append(p_topk)
        r_topks.append(r_topk)
        hit_topks.append(hit_topk)
        hit_at_ks.append(hit_at_k)
        mrrs.append(mrr)
        
    print("ndcg")
    print(np.mean(ndcgs))
    print("precision")
    print(np.mean(p_topks))
    print("recall")
    print(np.mean(r_topks))
    print("mrr")
    print(np.mean(mrrs))

    return ndcgs, p_topks, r_topks, hit_at_ks, mrrs
    


def biopathnet_performance(seed, threshold, pred_dir_list):

    base_path = os.path.join('reproduce', 'synleth', 'KR4SL_thr' + threshold)
    
    entities_path = os.path.join(base_path, 'data/transductive', 'entities.txt')
    entities = pd.read_csv(entities_path, sep=' ', header=None)
    entities.columns = ['entity', 'id', 'type']
    mapping = dict(entities[['entity', 'id']].values)
    
    test_q_path = os.path.join(base_path, 'results/trans_reason/epochs15_noDropout', f'test_q_{seed}.txt')
    test_q = pd.read_csv(test_q_path, sep=' ', header=None)
    test_q_lst = test_q[0].tolist() 
    
    gene_idx_path = os.path.join(base_path, 'results/trans_reason/epochs15_noDropout', f'gene_idx_{seed}.txt')
    gene_idx = pd.read_csv(gene_idx_path, sep=' ', header=None)
    gene_idx_lst = gene_idx[0].tolist() 
    
    o_all_path = os.path.join(base_path, 'results/trans_reason/epochs15_noDropout', f'o_all_seed_{seed}.npy')
    o_all = np.load(o_all_path, mmap_mode='r')
    
    s_all_path = os.path.join(base_path, 'results/trans_reason/epochs15_noDropout', f's_all_seed_{seed}.npy')
    s_all = np.load(s_all_path, mmap_mode='r')
    
    f_all_path = os.path.join(base_path, 'results/trans_reason/epochs15_noDropout', f'f_all_seed_{seed}.npy')
    f_all = np.load(f_all_path, mmap_mode='r')
    
    rm_file_path = os.path.join(base_path, 'results/trans_reason/epochs15_noDropout', f'r_all_seed_{seed}.txt')
    with open(rm_file_path, "r") as rm_file:
        rm = rm_file.read()
    r_all = rm.replace('\n', '.').split(".") 
    
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    ## BioPathNet performance
    for pred_dir in pred_dir_list:
        df = pd.read_csv(os.path.join('./experiments', pred_dir, 'predictions.csv'), sep='\t')
        
        ## 2. Create a new columns "gene1_id" and "gene2_id" by mapping gene symbols to index from entities.txt
        df['gene1_id'] = df.query_node.map(mapping)
        df['gene2_id'] = df.pred_node.map(mapping)
        
        ## 3. Reshape data from long to wide format
        df_long = df.filter(['gene1_id', 'gene2_id', 'probability'])
        df_wide = df_long.pivot_table(index='gene1_id', columns='gene2_id', values='probability', aggfunc='mean')
        
        ## 4. Reorder rows according to test_q
        df_wide = df_wide.reindex(test_q_lst)
        
        ## 5. Reorder columns according to gene_idx
        df_wide = df_wide.reindex(gene_idx_lst, axis=1)
        
        ## 6. Convert df with prediction scores to numpy array
        scores = df_wide.to_numpy()
        
        ## 7. Compute performance
        ndcg_1, p_1, r_1, hit_1, mrr_1 = cal_ndcg(scores, o_all, r_all, n=1)
        ndcg_3, p_3, r_3, hit_3, mrr_3 = cal_ndcg(scores, o_all, r_all, n=3)
        ndcg_5, p_5, r_5, hit_5, mrr_5 = cal_ndcg(scores, o_all, r_all, n=5)
        ndcg_10, p_10, r_10, hit_10, mrr_10 = cal_ndcg(scores, o_all, r_all, n=10)
        ndcg_20, p_20, r_20, hit_20, mrr_20 = cal_ndcg(scores, o_all, r_all, n=20)
        ndcg_50, p_50, r_50, hit_50, mrr_50 = cal_ndcg(scores, o_all, r_all, n=50)
        ndcg_100, p_100, r_100, hit_100, mrr_100 = cal_ndcg(scores, o_all, r_all, n=100)
        
        
        perf_dict = {'ndcg_1': ndcg_1, 'p_1': p_1, 'r_1': r_1, 'ndcg_3': ndcg_3, 'p_3': p_3, 'r_3': r_3, 'ndcg_5': ndcg_5, 'p_5': p_5, 'r_5': r_5, 'ndcg_10': ndcg_10, 'p_10': p_10, 'r_10': r_10, 'ndcg_20': ndcg_20, 'p_20': p_20, 'r_20': r_20, 'ndcg_50': ndcg_50, 'p_50': p_50, 'r_50': r_50, 'ndcg_100': ndcg_100, 'p_100': p_100, 'r_100': r_100, 'mrr_1': mrr_1, 'mrr_3': mrr_3, 'mrr_5': mrr_5, 'mrr_10': mrr_10, 'mrr_20': mrr_20, 'mrr_50': mrr_50, 'mrr_100': mrr_100} 
        
        perf_df = pd.concat([test_q, pd.DataFrame(perf_dict)], axis=1)
        
        perf_df.to_csv(os.path.join('./output', 'Performance_BioPathNet_perGene_threshold_' + threshold + '_seed_' + seed + '.csv'), sep=',')
        
        df_wide.to_csv(os.path.join('./output', 'Predictions_BioPathNet_threshold_' + threshold + '_seed_' + seed + '.csv'), sep=',')    
    

def kr4sl_performance(seed, threshold):
    # Base path for the files
    base_path = os.path.join('reproduce', 'synleth', 'KR4SL_thr' + threshold)
    
    # Load entities
    entities_path = os.path.join(base_path, 'data/transductive', 'entities.txt')
    entities = pd.read_csv(entities_path, sep=' ', header=None)
    entities.columns = ['entity', 'id', 'type']
    mapping = dict(entities[['entity', 'id']].values)
    
    # Load test queries
    test_q_path = os.path.join(base_path, 'results/trans_reason/epochs15_noDropout', f'test_q_{seed}.txt')
    test_q = pd.read_csv(test_q_path, sep=' ', header=None)
    test_q_lst = test_q[0].tolist()
    
    # Load gene indices
    gene_idx_path = os.path.join(base_path, 'results/trans_reason/epochs15_noDropout', f'gene_idx_{seed}.txt')
    gene_idx = pd.read_csv(gene_idx_path, sep=' ', header=None)
    gene_idx_lst = gene_idx[0].tolist()
    
    # Load prediction scores
    o_all_path = os.path.join(base_path, 'results/trans_reason/epochs15_noDropout', f'o_all_seed_{seed}.npy')
    o_all = np.load(o_all_path, mmap_mode='r')
    
    s_all_path = os.path.join(base_path, 'results/trans_reason/epochs15_noDropout', f's_all_seed_{seed}.npy')
    s_all = np.load(s_all_path, mmap_mode='r')
    
    f_all_path = os.path.join(base_path, 'results/trans_reason/epochs15_noDropout', f'f_all_seed_{seed}.npy')
    f_all = np.load(f_all_path, mmap_mode='r')
    
    # Load removed list
    rm_file_path = os.path.join(base_path, 'results/trans_reason/epochs15_noDropout', f'r_all_seed_{seed}.txt')
    with open(rm_file_path, "r") as rm_file:
        rm = rm_file.read()
    r_all = rm.replace('\n', '.').split(".")
    
    # Ensure output directory exists
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    ## KR4SL performance
    ndcg_1, p_1, r_1, hit_1, mrr_1 = cal_ndcg(s_all, o_all, r_all, n=1)
    ndcg_3, p_3, r_3, hit_3, mrr_3 = cal_ndcg(s_all, o_all, r_all, n=3)
    ndcg_5, p_5, r_5, hit_5, mrr_5 = cal_ndcg(s_all, o_all, r_all, n=5)
    ndcg_10, p_10, r_10, hit_10, mrr_10 = cal_ndcg(s_all, o_all, r_all, n=10)
    ndcg_20, p_20, r_20, hit_20, mrr_20 = cal_ndcg(s_all, o_all, r_all, n=20)
    ndcg_50, p_50, r_50, hit_50, mrr_50 = cal_ndcg(s_all, o_all, r_all, n=50)
    ndcg_100, p_100, r_100, hit_100, mrr_100 = cal_ndcg(s_all, o_all, r_all, n=100)
        
    perf_dict = {
        'ndcg_1': ndcg_1, 'p_1': p_1, 'r_1': r_1,
        'ndcg_3': ndcg_3, 'p_3': p_3, 'r_3': r_3,
        'ndcg_5': ndcg_5, 'p_5': p_5, 'r_5': r_5,
        'ndcg_10': ndcg_10, 'p_10': p_10, 'r_10': r_10,
        'ndcg_20': ndcg_20, 'p_20': p_20, 'r_20': r_20,
        'ndcg_50': ndcg_50, 'p_50': p_50, 'r_50': r_50,
        'ndcg_100': ndcg_100, 'p_100': p_100, 'r_100': r_100,
        'mrr_1': mrr_1, 'mrr_3': mrr_3, 'mrr_5': mrr_5,
        'mrr_10': mrr_10, 'mrr_20': mrr_20, 'mrr_50': mrr_50,
        'mrr_100': mrr_100
    }
    
    perf_df = pd.concat([test_q, pd.DataFrame(perf_dict)], axis=1)
    perf_df.to_csv(os.path.join('./output', f'Performance_KR4SL_perGene_threshold_{threshold}_seed_{seed}.csv'), sep=',')
    
    
    

## BioPathNet model performance

pred_dirs = ["2024-04-03-23-11-34-025040"] 
biopathnet_performance(seed = '1234', threshold = '030', pred_dir_list = pred_dirs)


## KR4SL model performance
kr4sl_performance(seed = '1234', threshold = '030')