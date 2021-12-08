#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Defending Graph Convolutional Networks against Dynamic Graph Perturbations via Bayesian Self-supervision
@topic: Employ GraphSS model to infer labels
@authors: Jun Zhuang, Mohammad Al Hasan.
"""

import sys
import time
import torch
import torch.nn.functional as F
from load_data import LoadDataset
from attack import select_target_nodes
from models_GCN import GCN
from models_SS import BayesSelfSupervision
from utils import read_pickle, gen_init_trans_matrix
from sklearn.metrics import accuracy_score

# ---Initialize the arugments---
try:    
    data_name = str(sys.argv[1])
    model_name = str(sys.argv[2])
    is_attacked = bool(sys.argv[3])
    TARGET_CLASS = int(sys.argv[4])
    attack_type = str(sys.argv[5])
    SAMPLE_RATE = float(sys.argv[6])
    GPU = int(sys.argv[7])
    is_inferable = bool(sys.argv[8]) # true / ""
    ssup_type = str(sys.argv[9])
    NUM_RETRAIN = int(sys.argv[10])
    NUM_INFER = int(sys.argv[11])
    WARMUP_STEP = int(sys.argv[12])
    Alert = bool(sys.argv[13]) # true / ""
    Dynamic_Phi = bool(sys.argv[14]) # true / ""
except:
    data_name = "cora" # cora, citeseer, pubmed, amazoncobuy, coauthor
    model_name = "GCN" # GCN, SGC, GraphSAGE
    is_attacked = True  # true or ""
    TARGET_CLASS = -1 # Target attack: class id; Non-target attack: -1.
    attack_type = "lf" # l, f, lf
    SAMPLE_RATE = 1.0 # The percentage of test nodes being attacked.
    GPU = -1 # -1 or [gpu_id]
    is_inferable = True # True / False
    ssup_type = "pseudo" # the type of self-supervising: "pseudo" or "noise"
    NUM_RETRAIN = 60 # the number of epochs for for retraining the node classifier
    NUM_INFER = 100 # the number of total epochs for inference
    WARMUP_STEP = 40 # the number of epochs for warm-up stage
    Alert = False # Alert or Recover
    Dynamic_Phi = True # Update the transition matrix or not
assert ssup_type == "pseudo" or ssup_type == "noise"
# The parameters for node classifier
LR = 0.001
N_LAYERS = 2
N_HIDDEN = 200
DROPOUT = 0
WEIGHT_DECAY = 0
if model_name == "GraphSAGE":
    aggregator_type = "gcn"
elif model_name == "GIN":
    aggregator_type = "sum" 
else:
    aggregator_type = None

# ---Preprocessing---
# Reload the labels and masks
label, Y_noisy, train_mask, val_mask, test_mask = \
                      read_pickle('../data/noisy_label/Y_gt_noisy_masks.pkl')
if is_attacked:
    # Load attacked graph
    print("Load the attacked graph on {0} dataset for {1}.".format(data_name, model_name))
    dirs_attack = '../data/attacker_data/'
    graph, feat, _, target_mask = read_pickle(dirs_attack+'G_atk_C{0}_T{1}_{2}_{3}.pkl' \
                                .format(TARGET_CLASS, attack_type, data_name, model_name))
    file_name = '_attack'
else:
    # Load clean graph
    data = LoadDataset(data_name)
    graph, feat, label = data.load_data()
    _, target_mask = select_target_nodes(label, test_mask, SAMPLE_RATE, atk_class=TARGET_CLASS)
    file_name = ''
#file_name = '_attack' if is_attacked else ''
# Reload the corresponding predicted labels
Y_pred, Y_pred_sm = read_pickle('../data/noisy_label/Y_preds{0}.pkl'.format(file_name))
# Reload the clean predicted labels for further self-supervision
Y_cpred, _ = read_pickle('../data/noisy_label/Y_preds.pkl')
# Convert tensor to numpy array
Y_gt, Y_cpred, Y_noisy, Y_pred, Y_pred_sm = \
    label.numpy(), Y_cpred.numpy(), Y_noisy.numpy(), Y_pred.numpy(), Y_pred_sm.detach().numpy()

# ---Initialize the initial warm-up transition matrix---
print("Initialize the warm-up transition matrix...")
Y_pred_sm_train = Y_pred_sm[train_mask] # predicted probability table (num_samples, num_classes)
Y_noisy_train = Y_noisy[train_mask] # noisy label
NUM_CLASSES = len(set(label.numpy()))
print("NUM_CLASSES: ", NUM_CLASSES)
TM_warmup = gen_init_trans_matrix(Y_pred_sm_train, Y_noisy_train, NUM_CLASSES)
print("The shape of warm-up TM is: ", TM_warmup.shape)

# ---Setup the gpu if necessary---
if GPU < 0:
    print("Using CPU!")
    cuda = False
else: # set GPU ref: https://pytorch.org/docs/stable/notes/cuda.html
    print("Using GPU!")
    cuda = True
    torch.cuda.set_device(GPU)
    graph = graph.to('cuda')
    feat = feat.cuda()
    train_mask = train_mask.cuda()
    val_mask = val_mask.cuda()

# ---Initialize the node classifier---
print("Initialize the node classifier...")
model = GCN(g=graph,
        in_feats=feat.shape[1],
        n_hidden=N_HIDDEN,
        n_classes=len(torch.unique(label)),
        n_layers=N_LAYERS, 
        activation=F.relu,
        dropout=DROPOUT,
        model_name=model_name,
        aggregator_type=aggregator_type)
if cuda: # if gpu is available
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# Path for saving the parameters
dirs = 'runs/{0}_{1}/'.format(data_name, model_name)
path = dirs + 'model_best.pth.tar'

# ---Bayesian Self-supervision---
if is_inferable:
    print("Infer the label...")
    Y_ssup_dict = {"pseudo":Y_cpred, "noise":Y_noisy}
    timer_0 = time.time()
    ss = BayesSelfSupervision(ALPHA=1.0) # Y_noisy or Y_cpred
    Y_infer, C_new = ss.infer_label(model, optimizer, dirs, graph, feat, train_mask, val_mask, \
                                    Y_pred, Y_pred_sm, Y_ssup_dict[ssup_type], Y_gt, TM_warmup, \
                                    NUM_RETRAIN, GPU, NUM_INFER, WARMUP_STEP, Alert, Dynamic_Phi)
    runtime = time.time() - timer_0
    print("\n Runtime: ", runtime)
    print("Y_inferred: \n {0} \n C_new: \n {1}".format(Y_infer, C_new))
# Evaluation after inference (accuracy)
print("Evaluation after inference:")
Y_infer, C_new = read_pickle('../data/noisy_label/Y_C.pkl') # array
# Evaluation on test/target graph
mask_dict = {"test_mask":test_mask, "target_mask":target_mask}
assert len(mask_dict) > 0
for mask_name, mask in mask_dict.items():
    Y_gt_mask = label[mask].numpy()
    Y_infer_mask = Y_infer[mask]
    acc_infer = accuracy_score(Y_gt_mask, Y_infer_mask)
    print("Accuracy of Y_infer on {0} graph: {1}.".format(mask_name, acc_infer))
