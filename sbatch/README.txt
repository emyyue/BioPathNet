README.txt

#################################################
###################### RUN ######################
#################################################

# to run NBFNet: make sure config file has the right data and output directory
# --gpus null for on CPU
# testing is done on "test.txt"

python script/run.py -s 1024 -c config/knowledge_graph/mock/mockdata_run.yaml --gpus [0] 

python script/run.py -s 1024 -c config/knowledge_graph/mock/mockdata_rotate.yaml --gpus [0]
python script/run.py -s 1024 -c config/knowledge_graph/mock/mockdata_transe.yaml --gpus [0] 
python script/run.py -s 1024 -c config/knowledge_graph/mock/mockdata_distmult.yaml --gpus [0] 
python script/run.py -s 1024 -c config/knowledge_graph/mock/mockdata_rgcn.yaml --gpus [0] 



##################################################
###################### PRED ######################
##################################################

# to do predictions given a checkpoint (model_epoch_8.pth)
# important predict.py will load test file named "test_pred.txt"
#### best way to generate this is to have all unique (head, relation, tail),
#### as predictions are done over p(all_t of node_type(t) | h, r), AS WELL AS p(all_h of node_type(h)| given t, r-1)
# important config file needs to have the flag "remove_pos: no"
#### else there will be no predictions of the specific triplets in test_pred.txt

python script/predict.py -c config/knowledge_graph/mock/mockdata_vis.yaml --gpus [0] --checkpoint dir/to/checkpoint/model_epoch_8.pth


#################################################
###################### VIS ######################
#################################################

# to visualize given a checkpoint (model_epoch_8.pth) and specific triplets to visualize in "test_vis.txt"
# same config file as in predictions can be used
# test_vis.txt needs to contain the ones to visualize
# also define a node_color_dict: which node type which color

## for graph
python script/visualize_graph.py -c config/knowledge_graph/mock/mockdata_vis.yaml --gpus [0] --checkpoint dir/to/checkpoint/model_epoch_8.pth

## for detail to each path
python script/visualize.py -c config/knowledge_graph/mock/mockdata_vis.yaml --gpus [0] --checkpoint dir/to/checkpoint/model_epoch_8.pth




#################################################
################## http-server ##################
#################################################
### visualize graph by following these instructions:
Instructions to run source stackoverflow: (https://stackoverflow.com/questions/38497334/how-to-run-html-file-on-localhost):
You can run your file in http-server.

1> Have Node.js installed in your system.

2> In CMD, run the command npm install http-server -g

3> Navigate to the specific path of your file folder in CMD and run the command http-server

4> Go to your browser and type localhost:8080. Your Application should run there.