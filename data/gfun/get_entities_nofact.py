kegg_genes = set()
all_pathways = set()
with open("./train2.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line[:-1]
        split = line.split('\t')
        all_pathways.add(split[0])
        kegg_genes.add(split[2])


with open("./entity_types_no_factgraph.txt", 'a') as f:
    for gene in kegg_genes:
        f.write(f'{gene}\t0\n')
    for pathway in all_pathways:
        f.write(f'{pathway}\t1\n')
