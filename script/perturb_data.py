import os
import pandas as pd
import numpy as np
import argparse

def perturb_graph(data_path, graph, perturbation_mode, k=0, seed=123):
    np.random.seed(seed)

    if graph == 'train1':
        df = pd.read_csv(os.path.join(data_path, 'train1.txt'), sep='\t', header=None)
    elif graph == 'train2':
        df = pd.read_csv(os.path.join(data_path, 'train2.txt'), sep='\t', header=None)
    else:
        raise ValueError("Undefined graph. Possible options are: train1, train2.")
    

    if perturbation_mode == 'remove_top_relations':
        relations_summary = df[1].value_counts()
        top_n_rels = relations_summary.nlargest(k).index
        df_perturbed = df[~df[1].isin(top_n_rels)]

    elif perturbation_mode == 'remove_top_kth_relation':
        relations_summary = df[1].value_counts()
        top_kth_rel = relations_summary.nlargest(k).index[-1]
        df_perturbed = df[df[1] != top_kth_rel]
    
    elif perturbation_mode == 'remove_random_relations':
        n_rm = round(len(df) * k / 100)
        df_perturbed = df.sample(len(df) - n_rm)
    
    elif perturbation_mode == 'add_random_relations':
        n_add = round(len(df) * k / 100)
        nodes_1 = df[0].unique()
        relations = df[1].unique()
        nodes_2 = df[2].unique()

        existing_edges = set()
        for row in df.itertuples(index=False):
            existing_edges.add((row[0], row[1], row[2]))
            existing_edges.add((row[2], row[1], row[0]))

        sampled_edges = []
        while len(sampled_edges) < n_add:
            new_edge = (np.random.choice(nodes_1), np.random.choice(relations), np.random.choice(nodes_2))
            if new_edge not in existing_edges and (new_edge[2], new_edge[1], new_edge[0]) not in existing_edges:
                sampled_edges.append(new_edge)
                existing_edges.add(new_edge)  
                existing_edges.add((new_edge[2], new_edge[1], new_edge[0]))
        
        new_edges_df = pd.DataFrame(sampled_edges, columns=[0, 1, 2])
        df_perturbed = pd.concat([df, new_edges_df]).drop_duplicates()
    
    elif perturbation_mode == 'remove_top_nodes':
        nodes_summary = pd.Series(df[0].tolist() + df[2].tolist()).value_counts()
        top_p_nodes = nodes_summary.nlargest(round(len(nodes_summary) * k / 100)).index
        df_perturbed = df[~df[0].isin(top_p_nodes) & ~df[2].isin(top_p_nodes)]

    else:
        raise ValueError("Undefined perturbation mode. Possible modes are: remove_top_relations, remove_top_kth_relation, remove_random_relations, add_random_relations, remove_top_nodes.")
    
    output_path = os.path.join(data_path, f'{graph}_{perturbation_mode}_{k}.txt')

    df_perturbed.to_csv(output_path, sep='\t', index=False, header=False)
    
    return df_perturbed


def input_arguments():
    parser = argparse.ArgumentParser(description='Perturb factual graph')
    parser.add_argument('--seed', type=int, default=123, help='Random state')
    parser.add_argument('--which_graph', type=str, default="train1", help='Graph to perform perturbations on')
    parser.add_argument('--data_path', type=str, default="/lustre/groups/crna01/projects/synthetic_lethality", help='Directory with input data')
    parser.add_argument('--perturbation_mode', type=str, default="remove_top_relations", help='Perturbation mode')
    parser.add_argument('--k', type=int, default=0, help='Percentage (number) of edges or node types or relation types to remove or add')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = input_arguments()
    perturbed_data = perturb_graph(data_path=args.data_path, graph=args.which_graph, perturbation_mode=args.perturbation_mode, k=args.k, seed=args.seed)
