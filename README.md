# Defending Graph Convolutional Networks against Dynamic Graph Perturbations via Bayesian Self-supervision

### Authors: Jun Zhuang, Mohammad Al Hasan.

### Paper:
Accepted by AAAI 2022'

### Abstract:
In recent years, plentiful evidence illustrates that Graph Convolutional Networks (GCNs) achieve extraordinary accomplishments on the node classification task. However, GCNs may be vulnerable to adversarial attacks on label-scarce dynamic graphs. Many existing works aim to strengthen the robustness of GCNs; for instance, adversarial training is used to shield GCNs against malicious perturbations. However, these works fail on dynamic graphs for which label scarcity is a pressing issue. To overcome label scarcity, self-training attempts to iteratively assign pseudo-labels to highly confident unlabeled nodes but such attempts may suffer serious degradation under dynamic graph perturbations. In this paper, we generalize noisy supervision as a kind of self-supervised learning method and then propose a novel Bayesian self-supervision model, namely GraphSS, to address the issue. Extensive experiments demonstrate that GraphSS can not only affirmatively alert the perturbations on dynamic graphs but also effectively recover the prediction of a node classifier when the graph is under such perturbations. These two advantages prove to be generalized over three classic GCNs across five public graph datasets.

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
ProGNN: [Graph Structure Learning for Robust Graph Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3394486.3403049) [[Code](https://github.com/ChandlerBang/Pro-GNN)] \
NRGNN: [NRGNN: Learning a Label Noise Resistant Graph Neural Network on Sparsely and Noisily Labeled Graphs](https://dl.acm.org/doi/abs/10.1145/3447548.3467364) [[Code](https://github.com/EnyanDai/NRGNN)]

### Cite
Please cite our paper if you think this repo is helpful.
```
@article{zhuang2022defending,
  title={Defending Graph Convolutional Networks against Dynamic Graph Perturbations via Bayesian Self-supervision},
  author={Zhuang, Jun and Al Hasan, Mohammad},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
