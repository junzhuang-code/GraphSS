# Defending Graph Convolutional Networks against Dynamic Graph Perturbations via Bayesian Self-supervision

### Authors: Anonymous

### Paper:
Under reviewed by AAAI 2022'

### Dataset:
 cora, citeseer, pubmed, amazoncobuy, coauthor

### Getting Started:
#### Prerequisites
 Linux or macOS \
 CPU or NVIDIA GPU + CUDA CuDNN \
 Python 3 \
 pytorch, dgl, numpy, scipy, sklearn, numba

#### Clone this repo
**git clone https://github.com/[user_name]/GraphSS.git** \
**cd GraphSS/graphSS**

#### Install dependencies
For pip users, please type the command: **pip install -r requirements.txt** \
For Conda users, you may create a new Conda environment using: **conda env create -f environment.yml**

#### Directories
##### graphSS:
 1. *pretrain.py*: Train the node classifier on the train graph
 2. *attack.py*: Implement adversarial attacks
 3. *main.py*: Infer labels by GraphSS
 4. *models_GCN.py*/*models_SS.py*: model scripts
 5. *load_data.py*: Load data script
 6. *utils.py*: Utils modules
##### data:
 1. *attacker_data*: a directory that stores attacked graph and GCN's weights
 2. *noisy_label*: a directory that stores labels

#### Runs
 1. Train the node classifier on the train graph \
  python [script_name] -data_name -model_name -NUM_EPOCHS -GPU -NOISY_RATIO -is_trainable \
  e.g. python pretrain.py cora GCN 200 0 0.1 true

 2. Apply adversarial attacks and generate predicted labels \
  python [script_name] -data_name -model_name -TARGET_CLASS -attack_type -NUM_PERT -SAMPLE_RATE -GPU -NOISY_RATIO -is_attack \
  e.g. python attack.py cora GCN -1 "lf" 2 0.2 0 0.1 true

 3. Inference and evaluation \
  python [script_name] -data_name -model_name -is_attacked -TARGET_CLASS -attack_type -SAMPLE_RATE -GPU -is_inferable -ssup_type -NUM_RETRAIN -NUM_INFER -WARMUP_STEP -Alert -Dynamic_Phi \
  e.g. python main.py cora GCN true -1 "lf" 0.1 0 true "pseudo" 60 100 40 "" true

 4. Visualization with Tensorboard \
  Under the directory of "GraphSS", run: Tensorboard --logdir=./runs/Logs_SS --port=8999

### Source of Competing Methods:
AdvTrain: [When Does Self-Supervision Help Graph Convolutional Networks?](http://proceedings.mlr.press/v119/you20a/you20a.pdf) [[Code](https://github.com/Shen-Lab/SS-GCNs)] \
GNN-Jaccard: [Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](https://www.ijcai.org/proceedings/2019/0669.pdf) [[Code](https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/defense/gcn_preprocess.py)] \
GNN-SVD: [All You Need Is Low (Rank): Defending Against Adversarial Attacks on Graphs](https://dl.acm.org/doi/pdf/10.1145/3336191.3371789) [[Code](https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/defense/gcn_preprocess.py)] \
RGCN: [Robust Graph Convolutional Networks Against Adversarial Attacks](https://dl.acm.org/doi/10.1145/3292500.3330851) [[Code](https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/defense/r_gcn.py)] \
GRAND: [Graph Random Neural Network for Semi-Supervised Learning on Graphs](https://arxiv.org/pdf/2005.11079.pdf) [[Code](https://github.com/THUDM/GRAND)] \
ProGNN: [Graph Structure Learning for Robust Graph Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3394486.3403049) [[Code](https://github.com/ChandlerBang/Pro-GNN)]
