import random

import numpy as np
import scipy.sparse as sp
import torch
import copy
import pickle
import math




'''
advice = 0
discussion = 1
friend = 2
'''
id_layer = 'advice'   # 要预测第几层

fname_lay = './data/moreno/CKM-Physicians-Innovation_layers.txt'
fname_adj = './data/moreno/CKM-Physicians-Innovation_multiplex.edges'
fname_features = './data/moreno/CKM-Physicians-Innovation_nodes.txt'

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

'''获取邻接矩阵，三维'''
def get_adj(fname, nodes_num, layer_size):
    #adj = torch.zeros(nodes_num,nodes_num) # 初始化邻接矩阵
    #layer_adj = torch.zeros(layer_size, nodes_num, nodes_num)
    layer_adj = np.zeros((layer_size, nodes_num, nodes_num))
    with open(fname, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            layer_adj[int(words[0])-1, int(words[1])-1, int(words[2])-1] = 1
    return layer_adj #

'''获取特诊矩阵，二维'''
def get_features(fname, nodes_num, features_num):
    features = torch.zeros(nodes_num, features_num)
    i = 0
    with open(fname, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            for i in range(len(words)-1):
                if words[0] != 'nodeID':
                    features[int(words[0])-1,i] = int(words[i+1])
    print(features.type)
    return features # 返回二维特征矩阵，type=Tensor
#get_features('./data/moreno/CKM-Physicians-Innovation_nodes.txt',246,13)

'''获取路径信息，二维list'''
def get_layer_info(fname):
    meta_path = []
    with open(fname, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            if words[0] != 'layerID':
                meta_path.append([words[0],words[1]])
    return meta_path # 返回路径信息，type=liat()->[[id,name]]
#get_layer_info('./data/moreno/CKM-Physicians-Innovation_layers.txt')

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
    Step4_adj = adj + np.eye(adj.shape[0])  # + adj.dot(adj) #+ adj.dot(adj).dot(adj) + adj.dot(adj).dot(adj).dot(adj)

    idx_test_positive = np.array(idx_test_list_positive)
    test_positive_num = idx_test_positive.shape[0]

    zero_location = list(np.where(np.array((Step4_adj + adj)).flatten() == 0))[0]
    temp = np.isin(zero_location, idx_train)
    idx_test_negative = np.random.choice(zero_location[np.where(temp == False)], size=test_positive_num, replace=False)

    idx_test = np.hstack((idx_test_positive, idx_test_negative))
    np.random.shuffle(idx_test)

    return idx_test

#可运行
def load_data_ckm_new_new(break_por=0.2):
    idx_features_labels = np.genfromtxt("{}{}.txt".format('./data/moreno/', 'CKM-Physicians-Innovation_nodes'),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels, dtype=np.float32)
    features = features[:,1:14]
    features = (np.array(features.todense()))
    labels = encode_onehot(idx_features_labels[:, -1])
    #np.save('./labels.npy', labels)
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.edges".format('./data/moreno/', 'CKM-Physicians-Innovation_multiplex'),
                                       dtype=np.int32)
    edges_unordered = edges_unordered[0:480,1:3]
    ## 2022/10/31
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = np.zeros((246,246))
    adj = adj + np.eye(246)
    for edge in edges:
        (i,j) = edge
        adj[i,j] = 1
        adj[j,i] =1
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # #adj = pickle.loads('adj.pkk')
    # with open('a', 'rb') as f:
    #     adj = pickle.load(f)
    # f.close()

    adj  = sp.csr_matrix(adj)

    #features = adj.todense()
    ##
    # adjs=np.zeros((3,246,246))
    # for i in range(edges_unordered.shape[0]):
    #     adjs[edges_unordered[i,0]-1,edges_unordered[i,1]-1,edges_unordered[i,2]-1] = 1
    # adj = sp.coo_matrix(adjs[0],dtype=np.float32)

    # adj = adjs[0]
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # 非对称矩阵转对称矩阵，有向图变无向图
    temp_ori_adj = copy.deepcopy(adj.todense())
    ori_adj = adj.todense()
    # adj = adj.todense()
    labels = (np.where(labels)[1])
    adj, idx_test_list_positive = random_breakLink(adj,break_por) #llb
    # adj, idx_test_list_positive = random_breakLink(adj, break_por)
    idx_train_noTensor = negative_sampling(adj,idx_test_list_positive)
    # adj = (ori_adj)
    # adj = np.ones((246,246))
    # for i in range(234):
    #     for j in range(234):
    #         adj[i,j]=0

    adj = (adj.todense())
    train_num = len(idx_train_noTensor)
    idx_train = (idx_train_noTensor[0:int(np.floor(train_num))])
    idx_val = (idx_train_noTensor[int(np.floor(train_num * 0.9)):])
    idx_test_list = test_negative_sampling(temp_ori_adj, idx_test_list_positive, idx_train_noTensor)
    idx_test = (idx_test_list)
    #print(idx_test)

    # adj = np.ones((246,246))
    # for i in range(226):
    #    for j in range(226):
    #         adj[i,j] = 0

    # temp added
    idx_train = idx_train_noTensor
    features = normalize(features)
    features = np.append(features, ori_adj, axis=1)
    print(ori_adj.shape, adj.shape, features.shape, idx_train.shape, idx_val.shape, idx_test.shape)
    print(ori_adj.dtype, adj.dtype, features.dtype, idx_train.dtype, idx_val.dtype, idx_test.dtype)
    print(ori_adj, adj, features, idx_train, idx_val, idx_test)
    return ori_adj, adj, features, idx_train, idx_val, idx_test




def load_data_ckm(break_por = 0.1):
    #dataACM = get_adj(fname_adj, 246, 3)
    # adjs = {
    #     "advice": dataACM[0],
    #     "discussion": dataACM[1],
    #     "friend": dataACM[2]
    # }
    # # 上述加载的是三条元路径
    #adj = dataACM[0]  # 现在获取id_layer层的原始邻接矩阵a

    layer_adj = np.ones((3, 246, 246))
    with open(fname_adj, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            layer_adj[int(words[0]) - 1, int(words[1]) - 1, int(words[2]) - 1] = 0
    #print(layer_adj[0,0,86])
    tempadj = layer_adj[0]
    a = list()
    for i in range(246):
        for j in range(246):
            if tempadj[i,j]==0:
                b=(i,j)
                a.append(b)

    adj = np.zeros((246,246))
    for (i,j) in a:
        adj[i][j]= 1
    print(adj[0,86],adj[0,89],adj[0,90])
    # for i in range(86):
    #    for j in range(96):
    #        adj[i][j] = 0
    #adj = np.where(adj==1,0,1)
    #adj = np.where(adj != 1,0,1)
    #adjTemp = layer_adj[0]
    #adjTemp = np.where(adjTemp!=1,0,1)
    # adj = layer_adj[0]
    # adj = np.where(adj==1,0,1)
    # print(adj)
    # adjTempx = layer_adj[0]
    # adjTemp = np.where(adjTempx==1,0,1)
    # print(adjTempx[0,86],adjTemp[0,86])


    # adj = np.ones((246,246))
    # for i in range(226):
    #    for j in range(226):
    #         adj[i,j] = 0
    temp_ori_adj = adj

    features = get_features(fname_features, 246, 13)
    features = np.array(features)
    features = normalize(features)
    #features = np.zeros((246,13))

    ori_adj = adj

    adj, idx_test_list_positive = random_breakLink(adj, break_por)
    idx_train_noTensor = negative_sampling(adj, idx_test_list_positive)
    train_num = len(idx_train_noTensor)

    idx_val = (idx_train_noTensor[int(np.floor(train_num * 0.9)):])
    idx_test_list = test_negative_sampling(temp_ori_adj, idx_test_list_positive, idx_train_noTensor)
    idx_test = (idx_test_list)

    idx_train = idx_train_noTensor

    #features = normalize(features)
    #features = np.zeros((246,1433))

    print(ori_adj.shape, adj.shape, features.shape, idx_train.shape, idx_val.shape, idx_test.shape)
    print(ori_adj.dtype, adj.dtype, features.dtype, idx_train.dtype, idx_val.dtype, idx_test.dtype)
    print(ori_adj, adj, features, idx_train, idx_val, idx_test)
    return ori_adj, adj, features, idx_train, idx_val, idx_test



def test_three():
    layer_adj = np.ones((3, 246, 246))
    with open(fname_adj, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            layer_adj[int(words[0]) - 1, int(words[1]) - 1, int(words[2]) - 1] = 0
    # print(layer_adj[0,0,86])
    tempadj = layer_adj[0]
    a = list()
    for i in range(246):
        for j in range(246):
            if tempadj[i, j] == 0:
                b = (i, j)
                a.append(b)

    adj = np.zeros((246, 246))
    for (i, j) in a:
        adj[i][j] = 1
    print(adj[0, 86], adj[0, 89], adj[0, 90])


def load_data_ckm_new(break_por=0.2):
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
    temp_adj = adjs[0]
    adj  = sp.csr_matrix(temp_adj)

    temp_ori_adj = copy.deepcopy(adj.todense())
    ori_adj = adj.todense()
    adj, idx_test_list_positive = random_breakLink(adj,break_por)
    idx_train_noTensor = negative_sampling(adj,idx_test_list_positive)

    adj = (adj.todense())
    train_num = len(idx_train_noTensor)
    idx_val = (idx_train_noTensor[int(np.floor(train_num * 0.9)):])
    idx_test_list = test_negative_sampling(temp_ori_adj, idx_test_list_positive, idx_train_noTensor)
    idx_test = (idx_test_list)

    # temp added
    idx_train = idx_train_noTensor
    features = normalize(features)
    features = np.append(features,temp_adj,axis=1)
    print(ori_adj.shape, adj.shape, features.shape, idx_train.shape, idx_val.shape, idx_test.shape)
    print(ori_adj.dtype, adj.dtype, features.dtype, idx_train.dtype, idx_val.dtype, idx_test.dtype)
    print(ori_adj, adj, features, idx_train, idx_val, idx_test)
    return ori_adj, adj, features, idx_train, idx_val, idx_test

'''获取多层邻接矩阵'''
def load_data_ckm_mutilayer(break_por=0.2):
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
    #ori_adj = adj.todense()
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


    return ori_adj, adj, features, idx_train, idx_train1, idx_train2, idx_val, idx_val1, idx_val2, idx_test, idx_test1, idx_test2
    # 预测多层
    # 原始邻接矩阵返回三层、break后的邻接矩阵返回三层、特征矩阵也返回三层、训练集返回三层、验证集返回三层、测试集返回三层