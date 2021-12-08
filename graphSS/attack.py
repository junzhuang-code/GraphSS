#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Defending Graph Convolutional Networks against Dynamic Graph Perturbations via Bayesian Self-supervision
@topic: Implement Nettack
@authors: Jun Zhuang, Mohammad Al Hasan.
@references:
    https://docs.dgl.ai/generated/dgl.DGLGraph.adjacency_matrix_scipy.html#dgl.DGLGraph.adjacency_matrix_scipy
"""

import sys
import numpy as np
import dgl
import torch
import torch.nn.functional as F
import scipy.sparse as ss
from models_GCN import GCN
from baselines.nettack import nettack as ntk
from load_data import LoadDataset
from utils import read_pickle, dump_pickle, split_masks, generate_random_noise_label
from pretrain import evaluation, prediction


def select_target_nodes(label, test_mask, sample_rate=0.1, atk_class=-1):
    """
    @topic: Select target nodes for targeted/non-targeted attack.
    @input:
        label (int tensor): ground-truth label;
        test_mask (bool tensor): the mask for testing set;
        sample_rate (float): the ratio of sampling in the testing set;
        atk_class (int): the attacked target class.
    @return:
        target_nodes_list (array): the list of target nodes;
        target_mask (bool tensor): the mask for target nodes.
    """
    target_mask = torch.zeros([len(label)], dtype=torch.bool)
    test_id_list = [i for i in range(len(test_mask)) if test_mask[i] == True]
    target_size = int(len(label[test_mask])*sample_rate) # Decide the size of target nodes
    if int(atk_class) in torch.unique(label): # Select "atk_class" nodes from test graph
        target_idx = [l for l in range(len(label)) if label[l] == atk_class and l in test_id_list]
        target_nodes_list = [i for i in target_idx[:target_size]]
    else: # Random select "target_size" nodes if "atk_class" doesn't belong to any existing classes
        np.random.seed(abs(atk_class)) # Fix the random seed for reproduction.
        #test_id_list = test_id_list[:int(target_size*1.2)]
        target_nodes_list = np.random.choice(test_id_list, target_size, replace=False)
    target_mask[target_nodes_list] = True # Generate the target mask
    return target_nodes_list, target_mask


def attack_graph(adj, feat, label, w0, w1, target_nodes, attack_type, num_pert=2):
    """
    @topic: Implement Nettack on target nodes.
    @input:
        adj/feat (sp.csr_matrix): Original adjacency matrix/feature matrix;
        label (list): The label of nodes;
        w0/w1 (list of list): The weights for input/output layer of GCN model;
        target_nodes (list): The target nodes (id \in [0, num_nodes-1]);
        attack_type (string): The type of attack (l/f/lf);
        num_pert (int): The number of perturbations.
    @return:
        graph_atk/feat_atk (dgl.graph/tensor): The attacked graph/feature matrix.
    """
    adj_, feat_ = adj, feat
    node_degrees = adj_.sum(axis=0).A1 # id \in [0, num_nodes-1]
    attack_type_list = ["l", "f", "lf"]
    if attack_type not in attack_type_list:
        print("Please select the correct type of attack!")
        return 0
    for node in target_nodes: # Traverse all targeted nodes
        if num_pert < 1:
            n_perturbation = int(node_degrees[node-1])
        else:
            n_perturbation = num_pert
        for _, atk_type in enumerate(attack_type):
            if atk_type == "l": # Select the type of attack
                perturb_structure, perturb_features = True, False
            if atk_type == "f":
                perturb_structure, perturb_features = False, True
                n_perturbation *= 10
            attacker = ntk.Nettack(adj_, feat_, label, w0, w1, node, verbose=True)
            attacker.reset()
            attacker.attack_surrogate(n_perturbation, perturb_structure, perturb_features) # direct attack
            adj_ = attacker.adj # Update the adj & feat iteratively
            feat_ = attacker.X_obs
    graph_atk = dgl.from_scipy(adj_) # Convert spmatrix to dgl graph (dgl 0.5.x)
    feat_atk = torch.FloatTensor(feat_.toarray())
    return graph_atk, feat_atk # Return the attacked graph/feat matrix


if __name__ == "__main__":    
    # ---Initialize the arugments---
    try:
        data_name = str(sys.argv[1])
        model_name = str(sys.argv[2])
        TARGET_CLASS = int(sys.argv[3]) # random seed = abs(TARGET_CLASS)
        attack_type = str(sys.argv[4])
        NUM_PERT = int(sys.argv[5])
        SAMPLE_RATE = float(sys.argv[6])
        GPU = int(sys.argv[7])
        NOISY_RATIO = float(sys.argv[8])
        is_attack = bool(sys.argv[9])
    except:
        data_name = "cora" # cora, citeseer, pubmed, amazoncobuy, coauthor
        model_name = "GCN" # GCN, SGC, GraphSAGE, TAGCN, GIN
        TARGET_CLASS = -1 # Target attack: class id; Non-target attack: -1, -2, ...
        attack_type = "lf" # l, f, lf
        NUM_PERT = 2 # Density of perturbations for target nodes.
        SAMPLE_RATE = 0.050 # The percentage of test nodes being attacked.
        GPU = -1 # -1 or [gpu_id]
        NOISY_RATIO = 0.1
        is_attack = False # true or "" for prediction
    dirs_attack = '../data/attacker_data/'
    CUT_RATE = 0.4
    # The paramaters for node classifier
    LR = 0.001
    N_LAYERS = 2
    N_HIDDEN = 200
    DROPOUT = 0
    WEIGHT_DECAY = 0
    if model_name == "GraphSAGE":
        #aggregator_type = "mean"
        aggregator_type = "gcn"
    elif model_name == "GIN":
        aggregator_type = "sum"
    else:
        aggregator_type = None

    # ---Preprocessing---
    # Load dataset
    data = LoadDataset(data_name)
    graph, feat, label = data.load_data()
    w0, w1 = read_pickle(dirs_attack+'W_{0}_{1}.pkl'.format(data_name, model_name))
    # Preprocessing the graph
    adj_sp = graph.adjacency_matrix(scipy_fmt="csr")
    feat_sp = ss.csr_matrix(feat)
    # Generate noisy label
    Y_noisy = generate_random_noise_label(label, noisy_ratio=NOISY_RATIO, seed=0)
    Y_noisy_lst = Y_noisy.tolist() # array --> list
    # Randomly split the train, validation, test mask by given cut rate
    val_mask, train_mask, test_mask = split_masks(label, cut_rate=CUT_RATE)
    # Present the average degree of test nodes
    node_degrees = adj_sp.sum(axis=0).A1
    print("The average degree of test nodes: ", np.mean(node_degrees[test_mask]))
    # Setup the gpu if necessary
    if GPU < 0:
        print("Using CPU!")
        cuda = False
    else:
        print("Using GPU!")
        cuda = True
        torch.cuda.set_device(GPU)
        graph = graph.to('cuda')
        feat = feat.cuda()
        label = label.cuda()
        test_mask = test_mask.cuda()

    # ---Initialize the model---
    print("Initialize the model...")
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
    path = 'runs/{0}_{1}/'.format(data_name, model_name) + 'model_best.pth.tar'

    if is_attack:
        # ---Attack on graph---
        # Implement node-level attack on the test graph (links, feats, and L&F)
        # If TARGET_CLASS = -1, non-target attack.
        print("Implement node-level attack on the test graph.")
        target_nodes_list, target_mask = \
            select_target_nodes(label, test_mask, SAMPLE_RATE, atk_class=TARGET_CLASS)
        target_mask = target_mask.cuda() if cuda else target_mask
        print("Evaluation before attack on target nodes: ")
        evaluation(model, optimizer, path, graph, feat, label, target_mask)
        graph_atk, feat_atk = attack_graph(adj_sp, feat_sp, Y_noisy_lst, w0, w1, \
                                         target_nodes_list, attack_type, NUM_PERT)
        graph_atk = graph_atk.to('cuda') if cuda else graph_atk
        feat_atk = feat_atk.cuda() if cuda else feat_atk
        graph_atk.ndata['feat'] = feat_atk # Update the attacked graph
        graph_atk.ndata['label'] = label
        print("Evaluation after attack on target nodes: ")
        evaluation(model, optimizer, path, graph_atk, feat_atk, label, target_mask)
        # Save the attacked graph
        if cuda:
            graph_atk, feat_atk, target_mask = graph_atk.cpu(), feat_atk.cpu(), target_mask.cpu()
        dump_pickle(dirs_attack+'G_atk_C{0}_T{1}_{2}_{3}.pkl' \
                    .format(TARGET_CLASS, attack_type, data_name, model_name), \
                    [graph_atk, feat_atk, target_nodes_list, target_mask])
    else:
        # ---Generate predicted label after attack---
        # If not attack, reload the attacked graph and mask.
        graph_atk, feat_atk, target_nodes_list, target_mask = \
                read_pickle(dirs_attack+'G_atk_C{0}_T{1}_{2}_{3}.pkl' \
                            .format(TARGET_CLASS, attack_type, data_name, model_name))
        graph_atk = graph_atk.to('cuda') if cuda else graph_atk
        feat_atk = feat_atk.cuda() if cuda else feat_atk
        print("Generate predicted label after attack.")
        Y_pred, Y_pred_sm = prediction(model, optimizer, path, graph_atk, feat_atk)
        if cuda:
            Y_pred, Y_pred_sm = Y_pred.cpu(), Y_pred_sm.cpu()
        dump_pickle('../data/noisy_label/Y_preds_attack.pkl', [Y_pred, Y_pred_sm])
        print("Y_pred/Y_pred_sm.shape: ", Y_pred.shape, Y_pred_sm.shape)
