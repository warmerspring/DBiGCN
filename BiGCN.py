import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from util_functions import dot_sim, use_cuda

''' 
Decomposed Graph Prototype Network (DGPN)
 --- At the firt layer, we decompose a k-hop gcn layer to {k+1} parts
 --- At the second-last layer, we use a fc layer to map the local and global embeddings to pred the csd_matrix. 
'''
device = use_cuda()


class BiGCN(nn.Module):
    def __init__(self, n_in, n_h, dropout):
        super(BiGCN, self).__init__()
        self.fc1 = nn.Linear(n_in, n_h, bias=True)
        # self.fc_local_pred_csd = nn.Linear(n_h, n_h, bias=True)
        # self.fc_final_pred_csd = nn.Linear(n_h, n_h, bias=True)  # used for the last layer

        self.dropout = dropout
        self.act = nn.ReLU()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

    def forward(self, X, S_X, S_A):
        # the local item: 1. get k-hop-gcn by one layer; 2. get local loss
        features = torch.mm(S_X, X)
        features = self.fc1(features)
        Y_X = torch.mm(features, S_A)
        # Y_X = self.act(features)
        Y_X = F.dropout(Y_X, p=self.dropout, training=self.training)
        return Y_X


class BiGCN_X(nn.Module):
    def __init__(self, n_in, n_h, dropout):
        super(BiGCN_X, self).__init__()
        self.fc1 = nn.Linear(n_in, 512, bias=True)
        self.fc2 = nn.Linear(512, n_h)
        # self.fc_local_pred_csd = nn.Linear(n_h, n_h, bias=True)
        # self.fc_final_pred_csd = nn.Linear(n_h, n_h, bias=True)  # used for the last layer

        self.dropout = dropout
        self.act = nn.ReLU()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

    #
    def forward(self, X, S_X, S_A):
        # the local item: 1. get k-hop-gcn by one layer; 2. get local loss
        features = torch.mm(S_X, X)
        features = self.fc1(features)
        features = self.act(features)
        features = self.fc2(features)
        Y_X = torch.mm(features, S_A)
        # Y_X = self.act(features)
        Y_X = F.dropout(Y_X, p=self.dropout, training=self.training)
        return Y_X
    # def forward(self, X, S_X, S_A):
    #     # the local item: 1. get k-hop-gcn by one layer; 2. get local loss
    #     features = self.fc1(X)
    #     features = self.act(features)
    #     features = torch.mm(S_X, features)
    #     features = self.fc2(features)
    #     Y_X = torch.mm(features, S_A)
    #     # Y_X = self.act(features)
    #     Y_X = F.dropout(Y_X, p=self.dropout, training=self.training)
    #     return Y_X
    # def forward(self, X, S_X, S_A):
    #     # the local item: 1. get k-hop-gcn by one layer; 2. get local loss
    #     features = self.fc1(X)
    #     features = self.act(features)
    #     features = self.fc2(features)
    #     features = self.act(features)
    #     features = torch.mm(S_X, features)
    #     Y_X = torch.mm(features, S_A)
    #     # Y_X = self.act(features)
    #     Y_X = F.dropout(Y_X, p=self.dropout, training=self.training)
    #     return Y_X


class BiGCN_A(nn.Module):
    def __init__(self, n_in, n_h, dropout):
        super(BiGCN_A, self).__init__()
        self.fc1 = nn.Linear(n_in, 800, bias=True)
        self.fc2 = nn.Linear(800, n_h, bias=True)
        # self.fc_local_pred_csd = nn.Linear(n_h, n_h, bias=True)
        # self.fc_final_pred_csd = nn.Linear(n_h, n_h, bias=True)  # used for the last layer

        self.dropout = dropout
        self.act = nn.ReLU()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

    def forward(self, X, S_X, S_A):
        # the local item: 1. get k-hop-gcn by one layer; 2. get local loss
        features = torch.mm(S_X, X)
        features = self.fc1(features)
        features = self.act(features)
        features = self.fc2(features)
        Y_X = torch.mm(features, S_A)
        # Y_X = self.act(features)
        Y_X = F.dropout(Y_X, p=self.dropout, training=self.training)
        return Y_X
