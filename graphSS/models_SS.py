#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Defending Graph Convolutional Networks against Dynamic Graph Perturbations via Bayesian Self-supervision
@topic: Model for Bayesian Self-supervision
@authors: Anonymous
"""

import os
import numpy as np
import torch
from pretrain import train, prediction
from utils import dump_pickle, tensor2array
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score


# Bayesian Self-supervision Model --------------------
class BayesSelfSupervision():
    def __init__(self, ALPHA=1.0):
        self.ALPHA = ALPHA # a parameter for computing TM (float).


    def get_labels_dict(self, Y_pred, Y_ssup):
        """
        @topic: Convert labels as dict
        @input: Y_pred/Y_ssup(1D-array).
        @return: infer_dict/ssup_dict(dict).
        """
        infer_dict = dict() # keys: idx; values: Y_pred.
        ssup_dict = dict() # keys: idx; values: Y_ssup. 
        idx = np.array([i for i in range(len(Y_ssup))])
        for i in range(len(idx)):
            infer_dict[idx[i]] = Y_pred[i]
            ssup_dict[idx[i]] = Y_ssup[i]
        return infer_dict, ssup_dict


    def generate_counting_matrix(self, Y_pred, Y_ssup, NUM_CLASSES):
        """
        @topic: Generate counting matrix and testing labels
        @input: Y_pred/Y_ssup (1D-array); NUM_CLASSES (int).
        @return: C (2D-array).
        """
        C_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for i in range(len(Y_ssup)):
            r = Y_pred[i]
            c = Y_ssup[i]
            C_matrix[r][c] += 1
        return C_matrix # (NUM_CLASSES, NUM_CLASSES)


    def approx_Gibbs_sampling(self, Y_pred_sm, Y_ssup, TM):
        """
        @topic: Approximate Gibbs Sampling
        @input:
            Y_pred_sm (2D Tensor: NUM_SAMPLES x NUM_CLASSES);
            Y_ssup (1D Tensor: NUM_SAMPLES x 1);
            TM (2D Tensor: NUM_CLASSES x NUM_CLASSES).
        @return: Y_infer (1D Tensor: NUM_SAMPLES x 1).
        """
        #unnorm_probs = preds * TM_ts.T[Y_pred_test_] # numpy (500, 1) * (500, 18) = (500, 18)
        unnorm_probs = Y_pred_sm * torch.index_select(torch.transpose(TM,0,1), 0, Y_ssup)
        probs = unnorm_probs / torch.sum(unnorm_probs, axis=1, keepdims=True)
        Y_infer = torch.max(probs, dim=1)[1]
        return Y_infer


    def infer_label(self, model, optimizer, dirs, graph, feat, train_mask, val_mask, \
                    Y_pred, Y_pred_sm, Y_ssup, Y_gt, TM_warmup, \
                    NUM_RETRAIN=50, GPU=-1, NUM_INFER=100, WARMUP_STEP=20, \
                    Alert=False, Dynamic_Phi=True):
        """
        @topic: Infer the labels with self-supervised labels (clean predicted label)
        @input:
            Y_pred/Y_ssup/Y_gt: predicted/self-supervised/groundtruth labels (1D array: NUM_SAMPLES x 1);
            Y_pred_sm: categorical distribution (2D array: NUM_SAMPLES x NUM_CLASSES);
            TM_warmup: warming-up transition matrix (2D array: NUM_CLASSES x NUM_CLASSES);
            NUM_RETRAIN: the number of epochs for retraining the node classifier (int);
            GPU: the ID of GPU device (int);
            NUM_INFER: the number of epochs for inference (int);
            WARMUP_STEP: using TM_warmup if step < WARMUP_STEP (int);
            Alert: Alert mode or not (bool);
            Dynamic_Phi: dynamically update the transition matrix or not (bool).
        @return:
            Y_inferred: new inferred labels (1D array: NUM_SAMPLES x 1);
            C: counting matrix (2D array: NUM_CLASSES x NUM_CLASSES).
        """
        # Get index of label
        idx = np.array([i for i in range(len(Y_ssup))])
        # Get Y_pred/Y_ssup dict
        z_dict, y_dict = self.get_labels_dict(Y_pred, Y_ssup)
        # Generate counting matrix
        C = self.generate_counting_matrix(Y_pred, Y_ssup, int(Y_pred_sm.shape[1]))
        # Convert to pytorch tensor
        Y_pred_sm = torch.FloatTensor(Y_pred_sm)
        Y_ssup = torch.LongTensor(Y_ssup)
        C = torch.FloatTensor(C)
        TM_warmup = torch.FloatTensor(TM_warmup)
        # Setup the GPU
        if GPU >= 0:
            Y_pred_sm = Y_pred_sm.cuda()
            Y_ssup = Y_ssup.cuda()
            C = C.cuda()
            TM_warmup = TM_warmup.cuda()    
        # Setup the interval
        if WARMUP_STEP >= 1000:
            interval = int(WARMUP_STEP//100)
        else:
            interval = 10
        # Record the data if necessary # ref: https://pytorch.org/docs/stable/tensorboard.html
        writer = SummaryWriter(log_dir=os.path.join("runs", 'Logs_SS'))
        path = dirs + 'model_best.pth.tar' # The path of model parameters

        acc_best = 0
        for step in range(NUM_INFER):
            # Update transition matrix TM for every n steps
            #if step >= interval and step % interval == 0:
            if step % interval == 0:
                TM_i = (C + self.ALPHA) / torch.sum(C + self.ALPHA, axis=1, keepdims=True)
                TM_i = TM_i.cuda() if GPU >= 0 else TM_i
                print(".", end = ' ')
                #print("In step {0}, TM is updated.".format(step))
            # Infer Z by Gibbs sampling based on corresponding TM
            if step < WARMUP_STEP:
                Y_infer = self.approx_Gibbs_sampling(Y_pred_sm, Y_ssup, TM_warmup)
            else:
                Y_infer = self.approx_Gibbs_sampling(Y_pred_sm, Y_ssup, TM_i)
            Y_infer = Y_infer.cuda() if GPU >= 0 else Y_infer
            # Update the counting matrix C
            if Dynamic_Phi:
                #print("Update the transition matrix")
                for num_i, idx_i in enumerate(idx):
                    C[z_dict[idx_i]][y_dict[idx_i]] -= 1
                    assert C[z_dict[idx_i]][y_dict[idx_i]] >= 0
                    z_dict[idx_i] = int(Y_infer[num_i])
                    C[z_dict[idx_i]][y_dict[idx_i]] += 1

            # Tensorboard --logdir=./runs/Logs_SS --port 8999
            if step % interval == 0:
                Y_infer_i = np.array([v for v in z_dict.values()])
                # Compute accuracy for every n steps
                acc_i = accuracy_score(Y_gt, Y_infer_i)
                print("Accuracy of Y_infer[i]: ", acc_i)
                writer.add_scalar('Accuracy_Y_infer', acc_i, step)
                #acc_i = accuracy_score(Y_ssup.numpy(), Y_infer_i)
                # Update the node classifier based on current z in given interval.
                if acc_i > acc_best and not Alert:
                    print("Update the model in step {0}:".format(step))
                    acc_best = acc_i
                    Y_infer_i = torch.LongTensor(Y_infer_i) 
                    Y_infer_i = Y_infer_i.cuda() if GPU >= 0 else Y_infer_i
                    train(model, optimizer, dirs, feat, Y_infer_i, \
                          train_mask, val_mask, NUM_RETRAIN)
                    _, Y_pred_sm = prediction(model, optimizer, path, graph, feat)
                    Y_pred_sm = Y_pred_sm.cuda() if GPU >= 0 else Y_pred_sm 

        # Get new infer label z
        Y_inferred = np.array([v for v in z_dict.values()])
        # Store the parameters
        C = tensor2array(C, GPU) # array
        dump_pickle('../data/noisy_label/Y_C.pkl', [Y_inferred, C])
        writer.close()
        return Y_inferred, C
