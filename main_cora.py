import os
import time

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from BiGCN import BiGCN, BiGCN_X, BiGCN_A
from util_functions import get_data_split, get_acc, setup_seed, use_cuda, cal_rbf_dist
from util_functions import load_data_set, symmetric_normalize_adj

device = use_cuda()
# device = torch.device('cuda:0')

# setup_seed(11)


def train(args):
    [c_train, c_val] = args.train_val_class
    idx, labellist, G, features, csd_matrix = load_data_set(args.dataset)
    csd_matrix_1nn_graph = cal_rbf_dist(data=csd_matrix.numpy(), n_neighbors=2, t=1.0)
    G = symmetric_normalize_adj(G).todense()
    csd_matrix_1nn_graph = symmetric_normalize_adj(csd_matrix_1nn_graph).todense()

    # my_feature_list = get_lrw_pre_calculated_feature_list(features, torch.FloatTensor(G), k=args.k, beta=args.beta)
    idx_train, idx_test, idx_val = get_data_split(c_train=c_train, c_val=c_val, idx=idx, labellist=labellist)
    y_true = np.array([int(temp[0]) for temp in labellist])  # [n, 1]
    y_true = torch.from_numpy(y_true).type(torch.LongTensor).to(device)

    num_sample = features.shape[0]
    num_class = torch.unique(y_true).shape[0]

    num_label_sample = len(idx_train)
    Y_true = torch.zeros([num_label_sample, num_class])
    for i1 in range(num_label_sample):
        Y_true[i1, y_true[i1]] = 1
    G = torch.tensor(data=G).to(device)
    csd_matrix_1nn_graph = torch.tensor(data=csd_matrix_1nn_graph, dtype=torch.float32).to(device)
    csd_matrix = csd_matrix.to(device)
    features = features.to(device)
    # model = DGPN(n_in=my_feature_list[0].shape[1], n_h=args.n_hidden, dropout=args.dropout).to(device)
    model_X = BiGCN_X(n_in=features.shape[1], n_h=num_class, dropout=args.dropout).to(device)
    model_A = BiGCN(n_in=csd_matrix.shape[1], n_h=num_sample, dropout=args.dropout).to(device)
    criterion_X = nn.CrossEntropyLoss()
    criterion_A = nn.CrossEntropyLoss()
    optimiser_X = torch.optim.Adam(model_X.parameters(), lr=args.lr, weight_decay=args.wd)
    optimiser_A = torch.optim.Adam(model_A.parameters(), lr=args.lr, weight_decay=args.wd)
    time_str = time.strftime('%Y-%m-%d=%H-%M-%S', time.localtime())
    os.makedirs(name='/E/DBiGCN/' + time_str + '/')
    result_dir = '/E/DBiGCN/' + time_str + '/'
    result_file = open(file=result_dir + 'DBiGCN_Cora.txt', mode='w')
    alpha_arr = np.array([0.01, 0.1, 0, 1, 10, 100])
    beta_arr = np.array([0.01, 0.1, 0, 1, 10, 100])

    for alpha in alpha_arr:
        for beta in beta_arr:
            for epoch in range(args.n_epochs + 1):
                model_X.train()
                model_A.train()
                optimiser_X.zero_grad()
                optimiser_A.zero_grad()

                Y_X = model_X(X=features, S_X=G, S_A=csd_matrix_1nn_graph)
                Y_A = model_A(X=csd_matrix, S_X=csd_matrix_1nn_graph, S_A=G)

                # cross entropy loss1

                loss_Y_X = criterion_X(Y_X[idx_train], y_true[idx_train])
                loss_Y_A = criterion_A(Y_A.T[idx_train], y_true[idx_train])
                Y_X = F.softmax(input=Y_X, dim=1)
                Y_A = F.softmax(input=Y_A, dim=0)
                diff1 = torch.mm(Y_X[idx_train, :], Y_A[:, idx_train])
                diff2 = torch.mm(Y_true, Y_true.T).to(device)
                loss_consistent = torch.norm(input=diff1 - diff2, p='fro')
                loss_consistent = loss_consistent * loss_consistent
                loss = loss_Y_X + alpha * loss_Y_A + beta * 0.000001 * loss_consistent
                # pres_X = np.argmax(Y_X.cpu().detach(), 1)
                # pres_A = np.argmax(Y_A.cpu().detach(), 1)
                loss.backward()
                optimiser_X.step()
                optimiser_A.step()
            model_X.eval()
            model_A.eval()
            Y_X = model_X(X=features, S_X=G, S_A=csd_matrix_1nn_graph)
            Y_A = model_A(X=csd_matrix, S_X=csd_matrix_1nn_graph, S_A=G)
            test_acc_X = get_acc(Y_X[idx_test], y_true[idx_test], c_train=c_train, c_val=c_val, model='test')
            test_acc_A = get_acc(Y_A.T[idx_test], y_true[idx_test], c_train=c_train, c_val=c_val, model='test')
            print('Evaluation!', 'alpha:', alpha, 'beta:', beta, 'Test_acc_X:', test_acc_X, 'Test_acc_A:', test_acc_A, "+++", file=result_file)
            result_file.flush()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MODEL')
    parser.add_argument("--dataset", type=str, default='cora', choices=['cora', 'citeseer', 'C-M10-M'],
                        help="dataset")
    parser.add_argument("--dropout", type=float, default=0, help="dropout probability")
    parser.add_argument("--train-val-class", type=int, nargs='*', default=[3, 0],
                        help="the first #train_class and #validation classes")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=10000, help="number of training epochs")
    # parser.add_argument("--n-hidden", type=int, default=128, help="number of hidden layers")
    parser.add_argument("--wd", type=float, default=0.0001, help="Weight for L2 loss")
    # parser.add_argument("--k", type=int, default=3, help="k-hop neighbors")
    # parser.add_argument("--beta", type=float, default=0.7,
    #                     help="probability of staying at the current node in a lazy random walk")
    # parser.add_argument("--alpha", type=float, default=1.0, help="hyper-parameter for local loss")
    args = parser.parse_args()
    print(args)
    train(args)
