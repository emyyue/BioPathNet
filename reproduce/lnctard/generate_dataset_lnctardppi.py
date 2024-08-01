import GTF
import pandas as pd
import networkx as nx
from pykeen.triples import TriplesFactory

# 1. split train2, valid and test

# 1.1. load corrected lnctard relations
lnctard = pd.read_csv("../silver/relations.csv", sep="\t", header=0, encoding="latin-1")
lnctard = lnctard[["Regulator", "SearchregulatoryMechanism", "Target"]].drop_duplicates()
# 1.2. use corrected gene name, filter out unmatched genes
genes_matched = pd.read_csv("../silver/genes_matched.csv", sep="\t", header=0, encoding="latin-1")
genes_matched = genes_matched[["gene_name_corrected", "gene_name_lnctard"]]
lnctard = pd.merge(lnctard, genes_matched, how="left", left_on="Regulator", right_on="gene_name_lnctard")
lnctard = pd.merge(lnctard, genes_matched, how="left", left_on="Target", right_on="gene_name_lnctard")
lnctard = lnctard[["gene_name_corrected_x", "SearchregulatoryMechanism", "gene_name_corrected_y"]].drop_duplicates()
lnctard = lnctard[lnctard["gene_name_corrected_x"].notnull() & lnctard["gene_name_corrected_y"].notnull()]
# 1.3. create graph
G_lnctard = nx.DiGraph()
for _, row in lnctard.iterrows():
  G_lnctard.add_edge(row['gene_name_corrected_x'], row['gene_name_corrected_y'], relation=row['SearchregulatoryMechanism'])
# 1.4. extract largest weakly connected component
largest_connected_component = max(nx.weakly_connected_components(G_lnctard), key=len)
G_lnctard_sub = G_lnctard.subgraph(largest_connected_component)
# 1.5. get triples in the weakly connected component
edges = [(u, G_lnctard_sub[u][v]['relation'], v) for u, v in G_lnctard_sub.edges()]
df_lnctard_sub = pd.DataFrame(edges, columns=['h', 'r', 't'])
# 1.6. split triples in to train2, valid and test using PyKeen
tf_ppi_sub = TriplesFactory.from_labeled_triples(df_lnctard_sub.to_numpy())
tf_train, tf_valid, tf_test = tf_ppi_sub.split([0.8, 0.1, 0.1], random_state=1234)
df_train = tf_train.tensor_to_df(tensor=tf_train.mapped_triples)[["head_label", "relation_label", "tail_label"]]
df_valid = tf_valid.tensor_to_df(tensor=tf_valid.mapped_triples)[["head_label", "relation_label", "tail_label"]]
df_test = tf_test.tensor_to_df(tensor=tf_test.mapped_triples)[["head_label", "relation_label", "tail_label"]]
# 1.7. save train2, valid and test
df_train.to_csv("../gold/lnctardppi/train2.txt", header=False, sep="\t", index=False)
df_valid.to_csv("../gold/lnctardppi/valid.txt", header=False, sep="\t", index=False)
df_test.to_csv("../gold/lnctardppi/test.txt", header=False, sep="\t", index=False)


# 2. generate train1 (ppi)

# 2.1. load ppi
ppi = pd.read_csv("../bronze/ppi.txt", sep=",", header=0)
# 2.2. load homosapiens
homosapiens = GTF.dataframe("../bronze/Homo_sapiens.GRCh38.110.gtf")
homosapiens = homosapiens[["gene_id", "gene_name", "gene_biotype"]].drop_duplicates()
# 2.3. find the protein coding genes and drop genes of other gene types
ppi_genes = pd.concat([ppi["h"], ppi["t"]], axis=0).drop_duplicates()
ppi_genes = pd.DataFrame({'gene_name': ppi_genes})
ppi_proteins = pd.merge(ppi_genes, homosapiens, on="gene_name", how="left")
ppi_proteins = ppi_proteins[ppi_proteins["gene_biotype"] == "protein_coding"]
ppi_proteins = ppi_proteins[["gene_name"]]
ppi = pd.merge(ppi, ppi_proteins, left_on="h", right_on="gene_name", how="inner")
ppi = pd.merge(ppi, ppi_proteins, left_on="t", right_on="gene_name", how="inner")
ppi = ppi[["h", "r", "t"]]
# 2.4. create graph
G_ppi = nx.DiGraph()
for _, row in ppi.iterrows():
    G_ppi.add_edge(row['h'], row['t'], relation=row['r'])
# 2.5. extract largest strongly connected component
largest_connected_component = max(nx.strongly_connected_components(G_ppi), key=len)
G_ppi_sub = G_ppi.subgraph(largest_connected_component)
# 2.6. get triples in the strongly connected component
edges_ppi = [(u, G_ppi_sub[u][v]['relation'], v) for u, v in G_ppi_sub.edges()]
df_ppi_sub = pd.DataFrame(edges_ppi, columns=['h', 'r', 't'])
# 2.7. save train1
df_ppi_sub.to_csv("../gold/lnctardppi/train1.txt", header=False, sep="\t", index=False)


# 3. generate entity_types

# 3.1. get lnctard genes and ppi genes
lnctard_genes = pd.concat([df_lnctard_sub["h"], df_lnctard_sub["t"]], axis=0).drop_duplicates()
lnctard_genes = pd.DataFrame({'gene': lnctard_genes})
ppi_genes = pd.concat([df_ppi_sub["h"], df_ppi_sub["t"]], axis=0).drop_duplicates()
ppi_genes = pd.DataFrame({'gene': ppi_genes})
# 3.2. lnctard gene types from genes_matched and create new type for ppi genes
lnctard_type = pd.merge(genes_matched, lnctard_genes, left_on="gene_name_corrected", right_on="gene", how="inner")
lnctard_type = lnctard_type[["gene_name_corrected", "gene_type_corrected"]].drop_duplicates()
ppi_type = ppi_genes.merge(lnctard_genes, on="gene", how='left', indicator=True).query('_merge == "left_only"').drop(columns='_merge')
ppi_type.rename(columns={'gene': 'gene_name_corrected'}, inplace=True)
ppi_type["gene_type_corrected"] = "protein_coding_ppi"
# 3.3. save entity_types
entity_types = pd.concat([lnctard_type, ppi_type], axis=0)
entity_types.to_csv("../gold/lnctardppi/entity_types.txt", header=False, sep="\t", index=False)

# 4. generate entity_names
entity_names = pd.concat([entity_types["gene_name_corrected"], entity_types["gene_name_corrected"]], axis=1)
entity_names.to_csv("../gold/lnctardppi/entity_names.txt", header=False, sep="\t", index=False)
