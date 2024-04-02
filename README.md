# NBFNet: Neural Bellman-Ford Networks #

This is the official codebase of the manuscript

BioKGC: Path-based reasoning in biomedical knowledge graphs

## Overview ##
BioKGC is a graph neural network framework, adapted from [NBFNet][paper],
designed to reason on biomedical knowledge graphs. BioKGC learns path representations
(instead of commonly used node embeddings) for the task of link prediction, 
specifically taking into account different node types and a 
background regulatory graph for message passing.


[paper]: https://arxiv.org/pdf/2106.06935.pdf

![BioKGC](asset/biokgc.svg)

This codebase is based on [NBFNetgithub], PyTorch and [TorchDrug]. It supports training and inference
with multiple GPUs or multiple machines.

[TorchDrug]: https://github.com/DeepGraphLearning/torchdrug
[NBFNetgithub]: https://github.com/DeepGraphLearning/NBFNet

## Installation ##

You may install the dependencies via either conda or pip. Generally, BioKGC works
with Python 3.7/3.8 and PyTorch version >= 1.8.0.

### From Conda ###

```bash
conda install torchdrug pytorch=1.8.2 cudatoolkit=11.1 -c milagraph -c pytorch-lts -c pyg -c conda-forge
conda install ogb easydict pyyaml -c conda-forge
```

### From Pip ###

```bash
pip install torch==1.8.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install torchdrug
pip install ogb easydict pyyaml
```
