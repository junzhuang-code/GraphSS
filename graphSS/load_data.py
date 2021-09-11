#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@title: Defending Graph Convolutional Networks against Dynamic Graph Perturbations via Bayesian Self-supervision
@topic: Load dataset
@authors: Anonymous
@references:
    https://docs.dgl.ai/api/python/data.html#citation-network-dataset
"""

from utils import preprocess_dgl_adj
import dgl.data

class LoadDataset():
    def __init__(self, data_name):
        self.data_name = data_name
        print("Current dataset: cora, citeseer, pubmed, amazoncobuy, coauthor.")
        print("Selecting {0} Dataset ...".format(self.data_name))

    def load_data(self):
        # Load dataset based on given data_name.
        if self.data_name == "cora": # cora_v2
            dataset = dgl.data.CoraGraphDataset()
        if self.data_name == "citeseer": # citeseer
            dataset = dgl.data.CiteseerGraphDataset()
        if self.data_name == "pubmed": # pubmed
            dataset = dgl.data.PubmedGraphDataset()
        if self.data_name == "amazoncobuy": # amazon_co_buy_photo
            dataset = dgl.data.AmazonCoBuyPhotoDataset()
        if self.data_name == "coauthor": # coauthor_cs
            dataset = dgl.data.CoauthorCSDataset()

        # Load graph, feature matrix, and label
        graph = dataset[0]
        feat = graph.ndata['feat'] # float32
        label = graph.ndata['label'] # int64

        # Preprocessing the adjacency matrix (dgl graph) and update the graph
        if self.data_name == "amazoncobuy" or self.data_name == "coauthor":
            graph = preprocess_dgl_adj(graph)
            graph.ndata['feat'] = feat
            graph.ndata['label'] = label

        print("Data is stored in: /Users/[user_name]/.dgl")
        print("{0} Dataset Loaded!".format(self.data_name))
        return graph, feat, label

"""
  #Graph:    cora,  citeseer, pubmed, amazoncobuy, coauthor
  #Nodes:    2708,  3327,     19717,  7650,        18333
  #Edges:    10556, 9228,     88651,  287326,      327576
  #Features: 1433,  3703,     500,    745,         6805
  #Classes:  7,     6,        3,      8,           15
"""
