from py2neo import Graph
import pandas as pd

genes = pd.read_csv("../silver/genes_matched.csv", sep="\t", header=0, encoding="latin-1")
genes = genes[["gene_name_lnctard", "gene_name_corrected", "gene_type_corrected"]]
relations = pd.read_csv("../silver/relations.csv", sep="\t", header=0, encoding="latin-1")
relations = pd.merge(relations, genes, how="left", left_on="Regulator", right_on="gene_name_lnctard")
relations = pd.merge(relations, genes, how="left", left_on="Target", right_on="gene_name_lnctard")
relations = relations[["gene_name_corrected_x", "SearchregulatoryMechanism", "gene_name_corrected_y"]].drop_duplicates()
relations = relations[relations["gene_name_corrected_x"].notnull() & relations["gene_name_corrected_y"].notnull()]
relations = relations.rename(columns={
    "gene_name_corrected_x": "Regulator",
    "gene_name_corrected_y": "Target"})
df = relations.drop_duplicates()

graph = Graph("bolt://localhost:7687", auth=("neo4j", "123qweasd"))

for _, row in df.iterrows():
    h, r, t = row["Regulator"], row["SearchregulatoryMechanism"], row["Target"]
    query = f"MERGE (h:RNA{{name:'{h}'}})\n" \
            f"MERGE (t:RNA{{name:'{t}'}})\n" \
            f"MERGE (h)-[r:Regulatory{{name:'{r}'}}]->(t)"
    graph.run(query)