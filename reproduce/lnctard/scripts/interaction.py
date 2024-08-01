import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv("../results/predictions_cancer_lncRNAs.csv", header=0)

df['prediction_node_rank'] = (
    df
    .groupby(["query_node", "query_node_degree", "query_relation"])["probability"]
    .rank(method='min', ascending=False)
)

avg_probability = (
    df
    .groupby(["query_relation", "prediction_node"])["probability"]
    .mean()
)
avg_probability = avg_probability.reset_index()
avg_probability = avg_probability.sort_values(by=["query_relation", "probability"], ascending=[True, False])
avg_probability['rank'] = (
    avg_probability
    .groupby(["query_relation"])["probability"]
    .rank(method='min', ascending=False)
)
avg_probability = avg_probability[avg_probability["rank"] <= 20]

gene_matched = pd.read_csv("../silver/genes_matched.csv", header=0, sep="\t")
avg_probability = pd.merge(avg_probability, gene_matched, left_on="prediction_node", right_on="gene_name_corrected", how="left")

relations = [
    "interact with mRNA",
    "transcriptional regulation",
    "interact with protein",
    "ceRNA or sponge",
    "expression association",
    "epigenetic regulation"
]

color_map = {
    "pseudogene": "#60A88D",
    "miRNA": "#3485A7",
    "lncRNA": "#8D3D7A",
    "protein_coding": "#F94C4F",
    "protein_coding_ppi": "#F47A38",
    "transcription_factor": "#F9CA59",

}

fig, axs = plt.subplots(2, 3, figsize=(18, 16))

for i, r in enumerate(relations):
    df = avg_probability[avg_probability["query_relation"] == r]
    plt.subplot(2, 3, i+1)
    for idx, row in df.iterrows():
        plt.barh(row['prediction_node'], row['probability'], color=color_map.get(row["gene_type_corrected"], '#ADD2C4'))
    plt.gca().invert_yaxis()
    plt.title(r)

plt.show()