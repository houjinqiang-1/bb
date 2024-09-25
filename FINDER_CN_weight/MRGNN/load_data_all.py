import random

import numpy as np
import scipy.sparse as sp
import torch
import copy
import pickle as pkl
import math
import scipy.sparse as sp
import scipy.io as scio


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot



'''矩阵归一化'''
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def random_breakLink(adj, break_por=0.1):
    idx_test_list = []
    N = np.shape(adj)[0]
    adj_new = copy.deepcopy(adj)
    cnt = 0;
    break_num = np.ceil(break_por * np.sum(adj) / 2)
    print('before_break_{}_break number{}'.format(np.sum(adj) / 2, break_num))
    while cnt < int(break_num):
        x_cor = np.random.randint(0, N - 1)
        y_cor = np.random.randint(0, N - 1)
        if adj_new[x_cor, y_cor] == 1 and np.sum(adj_new[x_cor, :]) != 1 and np.sum(adj_new[y_cor, :]) != 1:
            idx_test_list.append(x_cor * N + y_cor)
            idx_test_list.append(y_cor * N + x_cor)
            cnt += 1
            adj_new[x_cor, y_cor] = 0
            adj_new[y_cor, x_cor] = 0

    print('random_break finished')
    return adj_new, idx_test_list


def negative_sampling(adj_new, idx_test_list):
    # one positive combined with one negative, sample list for train
    # highOrder_adj represent nodes which have no connection in high order
    Step4_adj = adj_new  # +adj_new.dot(adj_new)#+adj_new.dot(adj_new).dot(adj_new)+adj_new.dot(adj_new).dot(adj_new).dot(adj_new)

    idx_train_positive = np.array(list(np.where(np.array(adj_new.todense()).flatten() == 1))[0])
    train_positive_num = idx_train_positive.shape[0]

    print(train_positive_num)
    zero_location = list(np.where(np.array(Step4_adj.todense()).flatten() == 0))[0]
    temp = np.isin(zero_location, idx_test_list)

    idx_train_negative = np.random.choice(zero_location[np.where(temp == False)], size=train_positive_num,
                                          replace=False)
    print(idx_train_negative.shape[0])
    idx_train = np.hstack((idx_train_negative, idx_train_positive))
    np.random.shuffle(idx_train)
    print(idx_train.shape[0])
    print('train negative sampling done')
    return idx_train


def test_negative_sampling(adj, idx_test_list_positive, idx_train):
#    Step4_adj = adj + np.eye(adj.shape[0])  # + adj.dot(adj) #+ adj.dot(adj).dot(adj) + adj.dot(adj).dot(adj).dot(adj)

    idx_test_positive = np.array(idx_test_list_positive)
    test_positive_num = idx_test_positive.shape[0]

    zero_location = list(np.where(np.array(adj).flatten() == 0))[0]
    temp = np.isin(zero_location, idx_train)
    idx_test_negative = np.random.choice(zero_location[np.where(temp == False)], size=test_positive_num, replace=False)

    idx_test = np.hstack((idx_test_positive, idx_test_negative))
    np.random.shuffle(idx_test)

    return idx_test

'''获取多层邻接矩阵'''
def load_data_ckm_mutilayer(break_por=0.1):
    idx_features_labels = np.genfromtxt("{}{}.txt".format('./data/moreno/', 'CKM-Physicians-Innovation_nodes'),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels, dtype=np.float32)
    features = features[:,1:14]
    features = (np.array(features.todense()))
    edges_unordered = np.genfromtxt("{}{}.edges".format('./data/moreno/', 'CKM-Physicians-Innovation_multiplex'),
                                       dtype=np.int32)
    adjs=np.zeros((3,246,246))
    for i in range(3):
        adjs[i] = adjs[i] + np.eye(246)
    for i in range(edges_unordered.shape[0]):
        adjs[edges_unordered[i,0]-1,edges_unordered[i,1]-1,edges_unordered[i,2]-1] = 1
        adjs[edges_unordered[i, 0] - 1, edges_unordered[i, 2] - 1, edges_unordered[i, 1] - 1] = 1

    temp_adj = adjs[0]
    adj  = sp.csr_matrix(temp_adj)
    temp_ori_adj = copy.deepcopy(adj.todense())
    ori_adj = adjs#改为多维邻接矩阵
    adj, idx_test_list_positive = random_breakLink(adj,break_por)
    idx_train_noTensor = negative_sampling(adj,idx_test_list_positive)

    adj = (adj.todense())
    train_num = len(idx_train_noTensor)
    idx_val = (idx_train_noTensor[int(np.floor(train_num * 0.9)):])
    idx_test_list = test_negative_sampling(temp_ori_adj, idx_test_list_positive, idx_train_noTensor)
    idx_test = (idx_test_list)

    # temp added
    idx_train = idx_train_noTensor
    features0 = normalize(features)
    features0 = np.append(features0,temp_adj,axis=1)
    '''第一层end'''


    '''第二层'''
    temp_adj1 = adjs[1] # 第二层         2022/11/17
    adj1 = sp.csr_matrix(temp_adj1) # 第二层         2022/11/17
    temp_ori_adj1 = copy.deepcopy(adj1.todense())
    adj1, idx_test_list_positive1 = random_breakLink(adj1, break_por)
    idx_train_noTensor1 = negative_sampling(adj1, idx_test_list_positive1)

    adj1 = (adj1.todense())
    train_num1 = len(idx_train_noTensor1)
    idx_val1 = (idx_train_noTensor1[int(np.floor(train_num1 * 0.9)):])
    idx_test_list1 = test_negative_sampling(temp_ori_adj1, idx_test_list_positive1, idx_train_noTensor1)
    idx_test1 = (idx_test_list1)

    idx_train1 = idx_train_noTensor1
    features1 = normalize(features)
    features1 = np.append(features1, temp_adj1, axis=1)
    '''end'''

    '''第三层'''
    temp_adj2 = adjs[2] # 第三层         2022/11/17
    adj2 = sp.csr_matrix(temp_adj2) # 第三层         2022/11/17
    temp_ori_adj2 = copy.deepcopy(adj2.todense())
    adj2, idx_test_list_positive2 = random_breakLink(adj2, break_por)
    idx_train_noTensor2 = negative_sampling(adj2, idx_test_list_positive2)

    adj2 = (adj2.todense())
    train_num2 = len(idx_train_noTensor2)
    idx_val2 = (idx_train_noTensor2[int(np.floor(train_num2 * 0.9)):])
    idx_test_list2 = test_negative_sampling(temp_ori_adj2, idx_test_list_positive2, idx_train_noTensor2)
    idx_test2 = (idx_test_list2)

    idx_train2 = idx_train_noTensor2
    features2 = normalize(features)
    features2 = np.append(features2, temp_adj2, axis=1)
    '''end'''

    '''三层合并'''
    adj = torch.stack([torch.tensor(adj), torch.tensor(adj1), torch.tensor(adj2)])
    features = np.zeros((3,features.shape[0],features.shape[1]+ori_adj.shape[1]))
    features[0] = features0;features[1] = features1;features[2] = features2
    '''end'''

    print(ori_adj.shape, adj.shape, features.shape, len(idx_train), len(idx_val), len(idx_test)
          , len(idx_train1), len(idx_val1), len(idx_test1),len(idx_train2), len(idx_val2), len(idx_test2))
    return ori_adj, adj, features, idx_train, idx_train1, idx_train2, idx_val, idx_val1, idx_val2, idx_test, idx_test1, idx_test2
    # 预测多层
    # 原始邻接矩阵返回三层、break后的邻接矩阵返回三层、特征矩阵也返回三层、训练集返回三层、验证集返回三层、测试集返回三层

def load_data_imdb():
    data = pkl.load(open('data/imdb.pkl', "rb"))
    ori_adj = np.zeros((2, 3550, 3550))
    ori_adj[0] = data['MDM']
    ori_adj[1] = data['MAM']

    features = data["feature"]
    features = sp.csr_matrix(features, dtype=np.float32)
    features = (np.array(features.todense()))

    '''第一层'''
    adj = sp.csr_matrix(ori_adj[0])
    temp_ori_adj = copy.deepcopy(adj.todense())
    adj, idx_test_list_positive = random_breakLink(adj, 0.1)
    idx_train_noTensor = negative_sampling(adj, idx_test_list_positive)
    adj = (adj.todense())
    train_num = len(idx_train_noTensor)
    idx_val = (idx_train_noTensor[int(np.floor(train_num * 0.9)):])
    idx_test_list = test_negative_sampling(temp_ori_adj, idx_test_list_positive, idx_train_noTensor)
    idx_test = (idx_test_list)
    idx_train = (idx_train_noTensor[:int(np.floor(train_num * 0.9))])
    '''第一层end'''

    '''第二层'''
    adj1 = sp.csr_matrix(ori_adj[1])
    temp_ori_adj1 = copy.deepcopy(adj1.todense())
    adj1, idx_test_list_positive1 = random_breakLink(adj1, 0.1)
    idx_train_noTensor1 = negative_sampling(adj1, idx_test_list_positive1)
    adj1 = (adj1.todense())
    train_num1 = len(idx_train_noTensor1)
    idx_val1 = (idx_train_noTensor1[int(np.floor(train_num1 * 0.9)):])
    idx_test_list1 = test_negative_sampling(temp_ori_adj1, idx_test_list_positive1, idx_train_noTensor1)
    idx_test1 = (idx_test_list1)
    idx_train1 = (idx_train_noTensor1[:int(np.floor(train_num1 * 0.9))])
    '''第二层end'''

    adj = torch.stack([torch.tensor(adj), torch.tensor(adj1)])  # , torch.tensor(adj2)])
    features_n = normalize(features)
    features = np.zeros((2, features.shape[0], features.shape[1] + ori_adj.shape[1]))
    features[0] = np.append(features_n, ori_adj[0], axis=1)
    features[1] = np.append(features_n, ori_adj[1], axis=1)

    print(ori_adj.shape, adj.shape, features.shape, len(idx_train), len(idx_val), len(idx_test)
          , len(idx_train1), len(idx_val1), len(idx_test1))
    return ori_adj, adj, features, idx_train, idx_train1, idx_val, idx_val1, idx_test, idx_test1

def load_data_amazon():
    data = pkl.load(open('data/amazon.pkl', "rb"))
    ori_adj = np.zeros((3, 7621, 7621))
    ori_adj[0] = data['IVI']
    ori_adj[1] = data['IBI']
    ori_adj[2] = data['IOI']

    features = data["feature"]
    features = sp.csr_matrix(features, dtype=np.float32)
    features = (np.array(features.todense()))
    '''第一层'''
    adj = sp.csr_matrix(ori_adj[0])
    temp_ori_adj = copy.deepcopy(adj.todense())
    adj, idx_test_list_positive = random_breakLink(adj, 0.1)
    idx_train_noTensor = negative_sampling(adj, idx_test_list_positive)
    adj = (adj.todense())
    train_num = len(idx_train_noTensor)
    idx_val = (idx_train_noTensor[int(np.floor(train_num * 0.9)):])
    idx_test_list = test_negative_sampling(temp_ori_adj, idx_test_list_positive, idx_train_noTensor)
    idx_test = (idx_test_list)
    idx_train = (idx_train_noTensor[:int(np.floor(train_num * 0.9))])
    '''第一层end'''

    '''第二层'''
    adj1 = sp.csr_matrix(ori_adj[1])
    temp_ori_adj1 = copy.deepcopy(adj1.todense())
    adj1, idx_test_list_positive1 = random_breakLink(adj1, 0.1)
    idx_train_noTensor1 = negative_sampling(adj1, idx_test_list_positive1)
    adj1 = (adj1.todense())
    train_num1 = len(idx_train_noTensor1)
    idx_val1 = (idx_train_noTensor1[int(np.floor(train_num1 * 0.9)):])
    idx_test_list1 = test_negative_sampling(temp_ori_adj1, idx_test_list_positive1, idx_train_noTensor1)
    idx_test1 = (idx_test_list1)
    idx_train1 = (idx_train_noTensor1[:int(np.floor(train_num1 * 0.9))])
    '''第二层end'''

    '''第三层'''
    adj2 = sp.csr_matrix(ori_adj[2])
    temp_ori_adj2 = copy.deepcopy(adj2.todense())
    adj2, idx_test_list_positive2 = random_breakLink(adj2, 0.1)
    idx_train_noTensor2 = negative_sampling(adj2, idx_test_list_positive2)
    adj2 = (adj2.todense())
    train_num2 = len(idx_train_noTensor2)
    idx_val2 = (idx_train_noTensor2[int(np.floor(train_num2 * 0.9)):])
    idx_test_list2 = test_negative_sampling(temp_ori_adj2, idx_test_list_positive2, idx_train_noTensor2)
    idx_test2 = (idx_test_list2)
    idx_train2 = (idx_train_noTensor2[:int(np.floor(train_num2 * 0.9))])
    '''第三层end'''

    adj = torch.stack([torch.tensor(adj), torch.tensor(adj1), torch.tensor(adj2)])
    features_n = normalize(features)
    features = np.zeros((3, features.shape[0], features.shape[1] + ori_adj.shape[1]))
    features[0] = np.append(features_n, ori_adj[0], axis=1)
    features[1] = np.append(features_n, ori_adj[1], axis=1)
    features[2] = np.append(features_n, ori_adj[2], axis=1)


    '''算法跑不动时随机采样'''
    # idx_test = np.random.choice(idx_test, size=int(len(idx_test)/2),replace=False)
    # idx_test1 = np.random.choice(idx_test1, size=int(len(idx_test1)/20),replace=False)
    # idx_train = np.random.choice(idx_train, size=int(len(idx_train) * 0.1), replace=False)
    # idx_val = np.random.choice(idx_val, size=int(len(idx_val) * 0.1), replace=False)
    # idx_test = np.random.choice(idx_test, size=int(len(idx_test) * 0.1), replace=False)
    # idx_train1 = np.random.choice(idx_train1, size=int(len(idx_train1) * 0.1), replace=False)
    # idx_val1 = np.random.choice(idx_val1, size=int(len(idx_val1) * 0.1), replace=False)
    # idx_test1 = np.random.choice(idx_test1, size=int(len(idx_test1) * 0.1), replace=False)
    ''''''

    print(ori_adj.shape, adj.shape, features.shape, len(idx_train), len(idx_val), len(idx_test)
          , len(idx_train1), len(idx_val1), len(idx_test1),len(idx_train2), len(idx_val2), len(idx_test2))
    return ori_adj, adj, features, idx_train, idx_train1, idx_train2, idx_val, idx_val1, idx_val2, idx_test, idx_test1, idx_test2

def load_data_acm():  #功能：获取features, adjs, break_adj, metapaths, train_idx,test_idx, train_label, test_label
    data = scio.loadmat("./data/ACM3025.mat")
    ori_adj = np.zeros((2,3025,3025))
    ori_adj[0] = data['PAP']
    ori_adj[1] = data['PLP']

    features = data["feature"]
    features = sp.csr_matrix(features, dtype=np.float32)
    features = (np.array(features.todense()))

    '''第一层'''
    adj = sp.csr_matrix(ori_adj[0])
    temp_ori_adj = copy.deepcopy(adj.todense())
    adj, idx_test_list_positive = random_breakLink(adj, 0.1)
    idx_train_noTensor = negative_sampling(adj, idx_test_list_positive)
    adj = (adj.todense())
    train_num = len(idx_train_noTensor)
    idx_val = (idx_train_noTensor[int(np.floor(train_num * 0.9)):])
    idx_test_list = test_negative_sampling(temp_ori_adj, idx_test_list_positive, idx_train_noTensor)
    idx_test = (idx_test_list)
    idx_train = (idx_train_noTensor[:int(np.floor(train_num * 0.9))])
    '''第一层end'''

    '''第二层'''
    adj1 = sp.csr_matrix(ori_adj[1])
    temp_ori_adj1 = copy.deepcopy(adj1.todense())
    adj1, idx_test_list_positive1 = random_breakLink(adj1, 0.1)
    idx_train_noTensor1 = negative_sampling(adj1, idx_test_list_positive1)
    adj1 = (adj1.todense())
    train_num1 = len(idx_train_noTensor1)
    idx_val1 = (idx_train_noTensor1[int(np.floor(train_num1 * 0.9)):])
    idx_test_list1 = test_negative_sampling(temp_ori_adj1, idx_test_list_positive1, idx_train_noTensor1)
    idx_test1 = (idx_test_list1)
    idx_train1 = (idx_train_noTensor1
[:int(np.floor(train_num1 * 0.9))])
    '''第二层end'''

    '''算法跑不动时随机采样'''
    # idx_train = np.random.choice(idx_train, size=int(len(idx_train) * 0.1), replace=False)
    # idx_val = np.random.choice(idx_val, size=int(len(idx_val) * 0.1), replace=False)
    # idx_test = np.random.choice(idx_test, size=int(len(idx_test) * 0.1), replace=False)
    # idx_train1 = np.random.choice(idx_train1, size=int(len(idx_train1) * 0.1), replace=False)
    # idx_val1 = np.random.choice(idx_val1, size=int(len(idx_val1) * 0.1), replace=False)
    # idx_test1 = np.random.choice(idx_test1, size=int(len(idx_test1) * 0.1), replace=False)
    ''''''

    adj = torch.stack([torch.tensor(adj), torch.tensor(adj1)])
    features_n = normalize(features)
    features = np.zeros((2, features.shape[0], features.shape[1] + ori_adj.shape[1]))
    features[0] = np.append(features_n, ori_adj[0], axis=1)
    features[1] = np.append(features_n, ori_adj[1], axis=1)

    print(ori_adj.shape, adj.shape, features.shape, len(idx_train), len(idx_val), len(idx_test)
          ,len(idx_train1), len(idx_val1), len(idx_test1))
    return ori_adj, adj, features, idx_train, idx_train1,idx_val, idx_val1,  idx_test, idx_test1
