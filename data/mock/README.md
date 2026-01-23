# BioPathNet Instructions

---

## Running BioPathNet

To run BioPathNet, ensure the config file has the correct data and output directory. Use `--gpus null` to run on CPU.

### Example Command
BioPathNet:
```bash
python script/run.py -s 1024 -c config/mock/mockdata_run.yaml --gpus [0]
````
Other models:
```bash
python script/run.py -s 1024 -c config/mock/mockdata_rotate.yaml --gpus [0]
python script/run.py -s 1024 -c config/mock/mockdata_transe.yaml --gpus [0]
python script/run.py -s 1024 -c config/mock/mockdata_distmult.yaml --gpus [0]
python script/run.py -s 1024 -c config/mock/mockdata_rgcn.yaml --gpus [0]
```

## Making Predictions

To run predictions using a model checkpoint (`model_epoch_8.pth`):

- **Test file**: `predict.py` will load `test_pred.txt` if not specified in config.
  - Best practice: include all unique `(head, relation, tail)` triplets.  
  - Predictions are computed as:
    - p(all t of node_type(t) | h, r)
    - p(all h of node_type(h) | given t, r-1)
- **Config file**: Make sure it has the flag `remove_pos: no`.  
  - Otherwise, no predictions will be generated for the triplets in `test_pred.txt`.

### Example Command
```bash
python script/predict.py \
    -c config/knowledge_graph/mock/mockdata_vis.yaml \
    --gpus [0] \
    --checkpoint dir/to/checkpoint/model_epoch_8.pth
```

## Visualize important paths using the interpretability


To visualize a model checkpoint (`model_epoch_8.pth`) with specific triplets:

- **Config file**: Use the same config file as used for predictions.
- **Triplets file**: `test_vis.txt` should contain the triplets you want to visualize.
- **Node colors**: Define a `node_color_dict` to map each node type to a color.

### Example Commands

```bash
## for graph
python script/visualize_graph.py -c config/mock/mockdata_vis.yaml --gpus [0] --checkpoint dir/to/checkpoint/model_epoch_8.pth

## for detail to each path
python script/visualize.py -c config/mock/mockdata_vis.yaml --gpus [0] --checkpoint dir/to/checkpoint/model_epoch_8.pth

```


## Visualize Graph via HTTP Server

Follow these steps to visualize your graph (output of `visualize_graph.py`) using a local HTTP server (based on [StackOverflow instructions](https://stackoverflow.com/questions/38497334/how-to-run-html-file-on-localhost)):

1. Install **Node.js** on your system if it’s not already installed.
2. Open CMD (Command Prompt) and run:

    ```bash
    npm install http-server -g
    ```

3. Navigate to the folder containing your files in CMD, then run:

    ```bash
    http-server
    ```

4. Open your browser and go to:

    ```
    localhost:8080
    ```

   Your application should now be running locally.
