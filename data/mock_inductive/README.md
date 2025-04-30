# Inductive Setting

In the inductive setting of graph link prediction, the model is trained on one set of nodes and edges 
but is expected to generalize to previously unseen nodes and edges during inference. 
This contrasts with the transductive setting, where all nodes are known during training.

## Data

Files:
- train1.txt
- train2.txt
- valid.txt
- test_graph.txt
- test.txt
- entity_names.txt
- entity_types.txt
- test_pred.txt


Training is performed on the triplets from train2.txt, while also using BRG from train1.txt for message passing.
The best model is selected based on the validation set (valid.txt). 
So far this is the exact procedure as in the transductive setting.
Then in inference time, the graph used to make predictions is replaced by test_graph.txt and the performance is measured
on the triplets from test.txt.

Entity files should contain all names and types from all of the files.

Prediction file test_pred.txt contains is a more concise version of test.txt, just as in the transductive case.

## Training and testing
```
python script/run.py --config config/mock/mockdata_inductive.yaml --gpus [0]
```

## Quick evaluation and prediction

````
python script/eval_and_predict_inductive.py -c config/mock/mockdata_inductive_pred.yaml --gpus [0]  --checkpoint /path/to/model_epoch_5.pth
````