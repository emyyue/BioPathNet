caner_related_lncrnas = [
    "NEAT1", "MALAT1", "SNHG16", "ZFAS1", "MIR22HG", "LINC-PINT", "MIR4435-2HG",
    "FGD5-AS1", "CYTOR", "PVT1", "FTX", "FOXCUT", "FENDRR", "LINC00461", "DLEU2",
    "MIR34AHG", "LINC00511", "LINC02582", "DLGAP1-AS1", "THORLNC"
]

enhancer_lncrnas = [
    "NEAT1", "ABALON", "MALAT1", "ASB16-AS1", "RAD51-AS1", "TNK2-AS1", "FBXL19-AS1",
    "ERICD", "ILF3-DT", "AGAP2-AS1", "SNHG8", "SOD2-OT1", "SNHG20", "MELTF-AS1",
    "RAB11B-AS1", "SNHG9"
]

crispr_lncrnas = ["CCAT1", "EPB41L4A-AS1", "LAMTOR5-AS1", "MIR17HG", "MIR210HG", "PVT1", "SNHG1", "SNHG12", "ZNF407-AS1"]


relations = [
    "interact with mRNA",
    "transcriptional regulation",
    "interact with protein",
    "ceRNA or sponge",
    "expression association",
    "epigenetic regulation"
]

tails = ["MALAT1", "MIR21", "TUG1", "TP73-AS1", "SCARNA13", "SNORD50A", "RN7SK", "MYC"]

heads = set(caner_related_lncrnas + enhancer_lncrnas + crispr_lncrnas)

with open('../gold/lnctardppi_pred/test.txt', 'w') as file:
    for h in heads:
        for r in relations:
            for t in tails:
                file.write(f"{h}\t{r}\t{t}\n")
