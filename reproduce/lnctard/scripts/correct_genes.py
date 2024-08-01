import numpy as np

import GTF
import pandas as pd
import networkx as nx


def gene_correction_map():
    # read Gencode
    gencode = GTF.dataframe("../bronze/gencode.gtf")
    gencode = gencode[["gene_id", "gene_name", "gene_type"]].drop_duplicates().rename(columns={
        "gene_name": "gene_name_gencode",
        "gene_type": "gene_type_gencode"
    })
    gencode['gene_id'] = gencode['gene_id'].str[:-2]
    gencode = gencode.drop_duplicates()

    # read Homo Sapiens GRCh38 110
    homosapiens = GTF.dataframe("../bronze/Homo_sapiens.GRCh38.110.gtf")
    homosapiens = homosapiens[["gene_id", "gene_name", "gene_biotype"]].drop_duplicates().rename(columns={
        "gene_name": "gene_name_homosapiens",
        "gene_biotype": "gene_type_homosapiens"
    })

    # read LncTarD 2.0
    lnctard = pd.read_csv("../bronze/lncTarD2.txt", sep="\t", header=0, encoding="latin-1")
    regulators = lnctard[["RegulatorEnsembleID", "RegulatorEntrezID", "Regulator", "RegulatorType"]].rename(columns={
        "RegulatorEnsembleID": "gene_id",
        "RegulatorEntrezID": "entrez_id",
        "Regulator": "gene_name_lnctard",
        "RegulatorType": "gene_type_lnctard"
    })
    targets = lnctard[["TargetEnsembleID", "TargetEntrezID", "Target", "TargetType"]].rename(columns={
        "TargetEnsembleID": "gene_id",
        "TargetEntrezID": "entrez_id",
        "Target": "gene_name_lnctard",
        "TargetType": "gene_type_lnctard"
    })
    genes = pd.concat([regulators, targets]).drop_duplicates()

    # join three tables
    genes = pd.merge(genes, gencode, how="left", on="gene_id")
    genes = pd.merge(genes, homosapiens, how="left", on="gene_id")
    genes.to_csv("../silver/genes_mapped.csv", header=True, sep="\t", encoding="latin-1", index=False)

    # get entrez id list
    genes_entrez = genes[(genes["gene_id"].isnull()) & (genes["entrez_id"].notnull())]
    print(genes_entrez.count())
    genes_entrez["entrez_id"].to_csv("../silver/entrezid.txt", header=False, index=False)

    # map aliases
    regulators = lnctard[["RegulatorEnsembleID", "RegulatorEntrezID", "Regulator", "RegulatorAliases"]].rename(columns={
        "RegulatorEnsembleID": "gene_id",
        "RegulatorEntrezID": "entrez_id",
        "Regulator": "gene_name_lnctard",
        "RegulatorAliases": "aliases"
    })
    targets = lnctard[["TargetEnsembleID", "TargetEntrezID", "Target", "TargetAliases"]].rename(columns={
        "TargetEnsembleID": "gene_id",
        "TargetEntrezID": "entrez_id",
        "Target": "gene_name_lnctard",
        "TargetAliases": "aliases"
    })
    genes = pd.concat([regulators, targets]).drop_duplicates()
    genes = genes[(genes["gene_id"].isnull())]
    genes = genes[genes["aliases"].notnull()][["gene_name_lnctard", "aliases"]]
    genes['aliases'] = genes['aliases'].str.split('|')
    names = genes.explode("aliases", ignore_index=True)
    homosapiens = homosapiens[["gene_id", "gene_name", "gene_biotype"]].drop_duplicates().rename(columns={
        "gene_name": "aliases",
        "gene_biotype": "gene_type_homosapiens"
    })
    names = pd.merge(names, homosapiens, on="aliases", how="left")
    print(names)


def gene_correction_match():
    # load and rename
    genes = pd.read_csv("../silver/genes_mapped.csv", sep="\t", header=0, encoding="latin-1")
    genes["gene_name_corrected"] = np.where(
        genes["gene_name_homosapiens"].notnull(), genes["gene_name_homosapiens"], genes["gene_name_gencode"]
    )
    genes["gene_type_corrected"] = np.where(
        genes["gene_type_homosapiens"].notnull(), genes["gene_type_homosapiens"], genes["gene_type_gencode"]
    )
    genes["gene_name"] = np.where(
        genes["gene_name_corrected"].notnull(), genes["gene_name_corrected"], genes["gene_name_lnctard"]
    )
    genes["gene_type"] = np.where(
        genes["gene_type_corrected"].notnull(), genes["gene_type_corrected"], genes["gene_type_lnctard"]
    )
    genes = genes[["gene_id", "gene_name_lnctard", "gene_type_lnctard", "gene_name_corrected", "gene_type_corrected"]]

    # graph
    lnctard = pd.read_csv("../bronze/lncTarD2.txt", sep="\t", header=0, encoding="latin-1")
    G = nx.DiGraph()
    for _, row in lnctard.iterrows():
      G.add_edge(row['Regulator'], row['Target'], relation=row['SearchregulatoryMechanism'])

    cc = max(nx.weakly_connected_components(G), key=len)
    G_sub = G.subgraph(cc)

    # get degrees and see if in largest CC
    node_data = []
    for node in G.nodes():
        node_name = node
        node_degree = G.degree[node]
        if node in G_sub:
            node_data.append((node_name, node_degree, True))
        else:
            node_data.append((node_name, node_degree, False))
    degrees = pd.DataFrame(node_data, columns=['gene_name_lnctard', 'degree', "weakly_connected"])

    genes = pd.merge(genes, degrees, on="gene_name_lnctard", how="inner")[
        ["gene_id", "degree", "weakly_connected", "gene_name_lnctard", "gene_name_corrected", "gene_type_lnctard", "gene_type_corrected"]]
    genes = genes.sort_values(by=['degree'], ascending=[False])
    genes.to_csv("../silver/genes_matched.csv", header=True, sep="\t", encoding="latin-1", index=False)


gene_correction_map()
# manual correction
gene_correction_match()
