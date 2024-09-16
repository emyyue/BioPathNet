#!/bin/bash


## Gene function prediction task 
wget -nv https://download.baderlab.org/PathwayCommons/PC2/v12/PathwayCommons12.All.hgnc.sif.gz
gunzip PathwayCommons12.All.hgnc.sif.gz

## Drug repurposing task
git clone https://github.com/mims-harvard/TxGNN

## Synthetic lethality task: 
git clone https://github.com/JieZheng-ShanghaiTech/KR4SL
# Data files used in this study: 
ls ./KR4SL/data/
ls ./KR4SL/data/transductive/

## LnRNA-target prediction task:
wget -nv https://lnctard.bio-database.com/downloadfile/lnctard2.0.zip