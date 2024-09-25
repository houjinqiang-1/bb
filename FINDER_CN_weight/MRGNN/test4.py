"""
@Time ： 2023/4/10 21:25
@Auth ： llb
@File ：model_n.py
@IDE ：PyCharm
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy
import numpy as np
import glob

import sys

from mutil_layer_weight import LayerNodeAttention_weight, Cosine_similarity, SemanticAttention, BitwiseMultipyLogis
from utils3025 import load_data_acm

# sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages')
import torch
import glob
print(torch.__version__)
from encoders import Encoder
from aggregators import MeanAggregator, LSTMAggregator, PoolAggregator
from utils import *
from sklearn.metrics import roc_auc_score, f1_score
from loadCKM import  load_data_ckm_mutilayer
from load_imdb import load_data_imdb
from load_amazon import load_data_amazon
from sample import *
from loadLazega import load_data_lazega_mutilayer,load_data_celegans_mutilayer
#from tensorboardX import SummaryWriter

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""
def identify_emb_dim(node_num):
    embedding_dims = []
    if node_num < 500:
        embedding_dims = list(range(2, node_num, 2))
        embedding_dims.insert(0, node_num)
    else:
        embedding_dims = list(range(2, 100, 2))
        embedding_dims.extend(range(100, 200, 10))
        embedding_dims.extend(range(200, 500, 60))
        embedding_dims.append(500)
    return embedding_dims


class SupervisedGraphSage(nn.Module):

    def __init__(self,fea_class_, enc1, enc2, enc3, embed_dim, lay_num, predict_layer=0):
        super(SupervisedGraphSage, self).__init__()
        #        self.enc = [enc1,enc2]#,enc3]
        self.enc1 = enc1
        self.enc2 = enc2
        self.enc3 = enc3
        self.criterion = nn.BCELoss()
        self.accuracy = accuracy
        self.embed_dim = embed_dim
        self.logis = LogisticRegression(enc1.embed_dim, 1)
        self.lay_num = lay_num
        self.predict_layer = predict_layer
        self.fea_class_ = fea_class_[self.predict_layer]

    def forward(self, nodes,is_test=False):
       # ''' 如果不考虑多层 '''
       # if self.predict_layer == 0:
       #     embeds = self.enc1(nodes)
       # elif self.predict_layer == 1:
       #     embeds = self.enc2(nodes)
       # elif self.predict_layer == 2:
       #     embeds = self.enc3(nodes)
       # else:
       #     print('has error: {}'.format(self.predict_layer))
       # predict = self.logis(embeds.t())
       # return predict


        #''' 考虑多层,权重为固定值 '''
        embeds1 = self.enc1(nodes)
        embeds2 = self.enc2(nodes)
        embeds3 = self.enc3(nodes)

        # semantic
       # lazega
       #  if self.predict_layer == 0:
       #      result = embeds1 + 0.5123*embeds2 + 0.4877*embeds3
       #  elif self.predict_layer == 1:
       #      result = embeds1 + 0.4259*embeds2 + 0.5741*embeds3
       #  elif self.predict_layer == 2:
       #      result = embeds1 + 0.3163*embeds2 + 0.6837*embeds3
       #  else:
       #      print('has error: {}'.format(self.predict_layer))

       #  ckm
        if self.predict_layer == 0:
            result = embeds1 + 0.4811 * embeds2 + 0.5189 * embeds3
        elif self.predict_layer == 1:
            result = embeds1 + 0.4092 * embeds2 + 0.5908 * embeds3
        elif self.predict_layer == 2:
            result = embeds1 + 0.5121 * embeds2 + 0.4879 * embeds3
        else:
            print('has error: {}'.format(self.predict_layer))

       #  celegans
       #  if self.predict_layer == 0:
       #      result = embeds1 + 0.509 * embeds2 + 0.491 * embeds3
       #  elif self.predict_layer == 1:
       #      result = embeds1 + 0.6076 * embeds2 + 0.3924 * embeds3
       #  elif self.predict_layer == 2:
       #      result = embeds1 + 0.4911 * embeds2 + 0.5089 * embeds3
       #  else:
       #      print('has error: {}'.format(self.predict_layer))

        # bitwise
        # if self.predict_layer == 0:
        #     result = embeds1 + 0.4943*embeds2 + 0.4057*embeds3
        # elif self.predict_layer == 1:
        #     result = embeds1 + 0.5045*embeds2 + 0.4955*embeds3
        # elif self.predict_layer == 2:
        #     result = embeds1 + 0.4944*embeds2 + 0.5056*embeds3
        # else:
        #     print('has error: {}'.format(self.predict_layer))

        predict = self.logis(result.t())
        return result.t(),predict


        ''' 考虑动态多层 '''
        # embeds1 = self.enc1(nodes)
        # embeds2 = self.enc2(nodes)
        # embeds3 = self.enc3(nodes)
        # embeds = [embeds1.t(), embeds2.t(), embeds3.t()]
        #
        # layerNodeAttention_weight = SemanticAttention(self.embed_dim, dropout=0.5, alpha=0.5,
        #                                                      metapath_number=self.lay_num,
        #                                                      layer_predict=self.predict_layer)
        # result,weight = layerNodeAttention_weight(embeds, nodes)
        # predict = self.logis(result)
        #
        # if is_test==True:
        #     return result,predict,weight
        #
        # return predict  ##预测第几层

    def loss(self, nodes, targets):  # epoch,order,emb_dim,epoches,dataname, times):
        _,predict = self.forward(nodes)
        return self.criterion(predict, targets)

    def acc(self, nodes, targets):
        _,predict = self.forward(nodes)
        return self.accuracy(predict, targets)

    def Auc(self, nodes, targets):
        _,predict = self.forward(nodes)
        return roc_auc_score(targets, predict.cpu().detach().numpy())
    def get_data(self,nodes):
        embeds,predict = self.forward(nodes,is_test=True)
        return embeds,predict

def run(emb_dim, times, dataname=" ", num_samples1=5, num_samples2=5, batch_num=256, break_por=0.1,epoches=2):
    np.random.seed(1)
    random.seed(1)
    if dataname == "ckm":
        ori_adj, adj, feat_data, idx_train, idx_train1, idx_train2, idx_val, idx_val1, idx_val2, idx_test, idx_test1, idx_test2 = [
            np.array(item) for item in load_data_ckm_mutilayer(break_por=0.2)]
    if dataname == "acm":
        ori_adj, adj, feat_data, idx_train, idx_train1, idx_val, idx_val1, idx_test, idx_test1, idx_train_classi, idx_val_classi, idx_test_classi, label_classi \
            = [np.array(item) for item in load_data_acm()]
    if dataname == "imdb":
        ori_adj, adj, feat_data, idx_train, idx_train1, idx_val, idx_val1, idx_test, idx_test1 = [np.array(item) for
                                                                                                  item in
                                                                                                  load_data_imdb()]
    if dataname == "amazon":
        ori_adj, adj, feat_data, idx_train, idx_train1, idx_train2, idx_val, idx_val1, idx_val2, idx_test, idx_test1, idx_test2  = [np.array(item) for item in load_data_amazon()]
    if dataname == "lazega":
        ori_adj, adj, feat_data, idx_train, idx_train1, idx_train2, idx_val, idx_val1, idx_val2, idx_test, idx_test1, idx_test2 = [
            np.array(item) for item in load_data_lazega_mutilayer(break_por=break_por)]
    if dataname == 'celegans':
        ori_adj, adj, feat_data, idx_train, idx_train1, idx_train2, idx_val, idx_val1, idx_val2, idx_test, idx_test1, idx_test2 = [
            np.array(item) for item in load_data_celegans_mutilayer(break_por=break_por)]

    '''modify start'''
    idx_train = [idx_train, idx_train1, idx_train2]
    idx_val = [idx_val, idx_val1, idx_val2]
    idx_test = [idx_test, idx_test1, idx_test2]
    print(adj.shape)

    break_adj = adj
    ori_adj_t = ori_adj
    lay_num = ori_adj.shape[0]

    adj = []
    ori_adj = []
    labels = []
    ori_labels = []
    for kk in range(lay_num):
        adj_temp = break_adj[kk]
        adj.append(adj_temp)
        ori_adj_temp = ori_adj_t[kk]
        ori_adj.append(ori_adj_temp)
        labels.append(torch.Tensor(adj_temp.flatten()))
        ori_labels.append(ori_adj_temp.flatten())
    '''modify end'''

    num_nodes = break_adj.shape[1]

    adj_lists = []
    for ki in range(lay_num):
        adj_lists_temp = defaultdict(set)
        [row, col] = np.where(break_adj[ki] == 1)
        for i in range(row.size):
            adj_lists_temp[row[i]].add(col[i])
            adj_lists_temp[col[i]].add(row[i])
        adj_lists.append(adj_lists_temp)
    '''modify end'''

    agg_one = list()
    agg_two = list()
    enc_one = list()
    enc_two = list()
    for kik in range(lay_num):
        features_temp = nn.Embedding(num_nodes, feat_data[kik].shape[1])
        features_temp.weight = nn.Parameter(torch.FloatTensor(feat_data[kik]), requires_grad=True)
        # features_temp.cuda()
        agg_one.append(MeanAggregator(features_temp, cuda=False))
        enc_one.append(
            Encoder(features_temp, feat_data[kik].shape[1], 128, adj_lists[kik], agg_one[kik],
                    gcn=True, cuda=False))
    ''''''
    agg_two.append(MeanAggregator(lambda nodes: enc_one[0](nodes).t(), cuda=False))
    agg_two.append(MeanAggregator(lambda nodes: enc_one[1](nodes).t(), cuda=False))
    agg_two.append(MeanAggregator(lambda nodes: enc_one[2](nodes).t(), cuda=False))
    ''''''
    for kki in range(len(enc_one)):
        enc_two.append(
            Encoder(lambda nodes2: enc_one[kki](nodes2).t(), enc_one[kki].embed_dim, emb_dim,
                    adj_lists[kki], agg_two[kki],
                    base_model=enc_one[kki],
                    gcn=True, cuda=False))
        enc_one[kki].num_sample = num_samples1
        enc_two[kki].num_sample = num_samples2
    graphsage = []
    optimizer = []
    fea_class_ = list()
#    n_gpu = torch.cuda.device_count()
#    device_ids = list(range(n_gpu))
    for kn in range(lay_num):
        fea_class_.append(torch.randn(num_nodes,emb_dim,dtype=torch.float).cpu())
        graphsage_temp = SupervisedGraphSage(fea_class_,enc_two[0], enc_two[1], enc_two[2], emb_dim, lay_num,
                                             predict_layer=kn)
        graphsage.append(graphsage_temp)#.module)
        optimizer_temp = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage[kn].parameters()), lr=0.7)
        optimizer.append(optimizer_temp)
    '''modify end'''
#     return_list = iterate_return(idx_train[0], [10, 10], labels[0], adj[0], batch_num)  # 预测第一层
#     return_list1 = iterate_return(idx_train[1], [10, 10], labels[1], adj[1], batch_num)  # 预测第二层
#     return_list2 = iterate_return(idx_train[2], [10, 10], labels[2], adj[2], batch_num)  # 预测第三层
#
#     patience = 0
#     best_test = 0
#     best_epoch = 0
#     for epoch in range(epoches):
#         epoche_times = []
#         epoche_loss = []
#         epoche_loss_val = []
#         epoche_acc = []
#         epoche_acc_val = []
#         epoche_auc = []
#
#         '''2023/3/24 start'''
#         return_list_all = [return_list, return_list1, return_list2]
#         entity_dict = {'0': int(len(return_list)), '1': int(len(return_list1)), '2': int(len(return_list2))}
#         entity_dict = sorted(entity_dict.items(), key=lambda x: x[1])
#         entity_dict_sort = [int(x) for x in list(dict(entity_dict).keys())]
#         # print(len(return_list_all[entity_dict_sort[-1]]))
#         for kia in range(len(return_list_all[entity_dict_sort[-1]])):
#             loss = []
#             val_loss = []
#             acc = []
#             auc = []
#             val_acc = []
#             if kia < len(return_list_all[entity_dict_sort[0]]):
#                 train_nodes, train_targets, val_nodes, val_targets = return_list[kia]
#                 train_nodes1, train_targets1, val_nodes1, val_targets1 = return_list1[kia]
#                 train_nodes2, train_targets2, val_nodes2, val_targets2 = return_list2[kia]
#                 train_nodes = [int(node) for node in train_nodes]
#                 val_nodes = [int(node) for node in val_nodes]
#                 train_nodes1 = [int(node1) for node1 in train_nodes1]
#                 val_nodes1 = [int(node1) for node1 in val_nodes1]
#                 train_nodes2 = [int(node2) for node2 in train_nodes2]
#                 val_nodes2 = [int(node2) for node2 in val_nodes2]
#                 start_time = time.time()
#                 for ka1 in range(lay_num):
#                     optimizer[ka1].zero_grad()
#                 train_nodes = [train_nodes, train_nodes1, train_nodes2]
#                 train_targets = [train_targets, train_targets1, train_targets2]
#                 val_nodes = [val_nodes, val_nodes1, val_nodes2]
#                 val_targets = [val_targets, val_targets1, val_targets2]
#                 for km1 in range(lay_num):
#                     loss_t = graphsage[km1].loss(train_nodes[km1], train_targets[km1])
#                     val_loss_t = graphsage[km1].loss(val_nodes[km1], val_targets[km1])
#                     acc_t = graphsage[km1].acc(train_nodes[km1], train_targets[km1])
#                     auc_t = graphsage[km1].Auc(train_nodes[km1], train_targets[km1])
#                     val_acc_t = graphsage[km1].acc(val_nodes[km1], val_targets[km1])
#                     loss.append(loss_t)
#                     val_loss.append(val_loss_t)
#                     acc.append(acc_t)
#                     auc.append(auc_t)
#                     val_acc.append(val_acc_t)
#                 loss = (sum(loss)) / lay_num
#                 val_loss = (sum(val_loss)) / lay_num
#                 acc = (sum(acc)) / lay_num
#                 auc = (sum(auc)) / lay_num
#                 val_acc = (sum(val_acc)) / lay_num
#
#                 loss.backward()
#                 for kh1 in range(lay_num):
#                     optimizer[kh1].step()
#                 end_time = time.time()
#
#                 epoche_times.append(end_time - start_time)
#                 epoche_loss.append(loss)
#                 epoche_loss_val.append(val_loss)
#                 epoche_acc.append(acc)
#                 epoche_acc_val.append(val_acc)
#                 epoche_auc.append(auc)
#             elif kia < len(return_list_all[entity_dict_sort[1]]):
#                 train_nodes, train_targets, val_nodes, val_targets = return_list_all[entity_dict_sort[1]][kia]
#                 train_nodes1, train_targets1, val_nodes1, val_targets1 = return_list_all[entity_dict_sort[-1]][kia]
#                 train_nodes = [int(node) for node in train_nodes]
#                 val_nodes = [int(node) for node in val_nodes]
#                 train_nodes1 = [int(node1) for node1 in train_nodes1]
#                 val_nodes1 = [int(node1) for node1 in val_nodes1]
#                 start_time = time.time()
#                 for ka2 in range(lay_num):
#                     if ka2 == entity_dict_sort[0]:
#                         continue
#                     optimizer[ka2].zero_grad()
#                 train_nodes = [list(),train_nodes, train_nodes1]
#                 train_targets = [list(),train_targets, train_targets1]
#                 val_nodes = [list(),val_nodes, val_nodes1]
#                 val_targets = [list(),val_targets, val_targets1]
#                 kindex = 0
#                 for km2 in range(len(entity_dict_sort)):
#                     if km2 == 0:
#                         kindex += 1
#                         continue
#                     loss_t = graphsage[entity_dict_sort[km2]].loss(train_nodes[kindex], train_targets[kindex])
#                     val_loss_t = graphsage[entity_dict_sort[km2]].loss(val_nodes[kindex], val_targets[kindex])
#                     acc_t = graphsage[entity_dict_sort[km2]].acc(train_nodes[kindex], train_targets[kindex])
#                     auc_t = graphsage[entity_dict_sort[km2]].Auc(train_nodes[kindex], train_targets[kindex])
#                     val_acc_t = graphsage[entity_dict_sort[km2]].acc(val_nodes[kindex], val_targets[kindex])
#                     loss.append(loss_t)
#                     val_loss.append(val_loss_t)
#                     acc.append(acc_t)
#                     auc.append(auc_t)
#                     val_acc.append(val_acc_t)
#                     kindex += 1
#
#                 loss = (sum(loss)) / (lay_num - 1)
#                 val_loss = (sum(val_loss)) / (lay_num - 1)
#                 acc = (sum(acc)) / (lay_num - 1)
#                 auc = (sum(auc)) / (lay_num - 1)
#                 val_acc = (sum(val_acc)) / (lay_num - 1)
#
#                 loss.backward()
#                 for kh2 in range(lay_num):
#                     if kh2 == entity_dict_sort[0]:
#                         continue
#                     optimizer[kh2].step()
#                 end_time = time.time()
#
#                 epoche_times.append(end_time - start_time)
#                 epoche_loss.append(loss)
#                 epoche_loss_val.append(val_loss)
#                 epoche_acc.append(acc)
#                 epoche_acc_val.append(val_acc)
#                 epoche_auc.append(auc)
#             elif kia < len(return_list_all[entity_dict_sort[2]]):
#                 train_nodes, train_targets, val_nodes, val_targets = return_list_all[entity_dict_sort[-1]][kia]
#                 train_nodes = [int(node) for node in train_nodes]
#                 val_nodes = [int(node) for node in val_nodes]
#                 start_time = time.time()
#                 optimizer[entity_dict_sort[-1]].zero_grad()
#                 loss_t = graphsage[entity_dict_sort[-1]].loss(train_nodes, train_targets)
#                 val_loss_t = graphsage[entity_dict_sort[-1]].loss(val_nodes, val_targets)
#                 acc_t = graphsage[entity_dict_sort[-1]].acc(train_nodes, train_targets)
#                 auc_t = graphsage[entity_dict_sort[-1]].Auc(train_nodes, train_targets)
#                 val_acc_t = graphsage[entity_dict_sort[-1]].acc(val_nodes, val_targets)
#                 loss.append(loss_t)
#                 val_loss.append(val_loss_t)
#                 acc.append(acc_t)
#                 auc.append(auc_t)
#                 val_acc.append(val_acc_t)
#                 loss = (sum(loss)) / 1
#                 val_loss = (sum(val_loss)) / 1
#                 acc = (sum(acc)) / 1
#                 auc = (sum(auc)) / 1
#                 val_acc = (sum(val_acc)) / 1
#                 loss.backward()
#                 optimizer[entity_dict_sort[-1]].step()
#                 end_time = time.time()
#
#                 epoche_times.append(end_time - start_time)
#                 epoche_loss.append(loss)
#                 epoche_loss_val.append(val_loss)
#                 epoche_acc.append(acc)
#                 epoche_acc_val.append(val_acc)
#                 epoche_auc.append(auc)
#         print("epoches: {:04d} | ".format(epoch + 1),
#               "train loss: {:.4f} | ".format(sum(epoche_loss) / len(epoche_loss)),
#               "train acc: {:.4f} | ".format(sum(epoche_acc) / len(epoche_acc)),
#               "val loss: {:.4f} | ".format(sum(epoche_loss_val) / len(epoche_loss_val)),
#               "val acc: {:.4f} | ".format(sum(epoche_acc_val) / len(epoche_acc_val)),
#               "auc: {:.4f} | ".format(sum(epoche_auc) / len(epoche_auc)),
#               "time: {:.4f} | ".format(sum(epoche_times)))
#         ## 2022/10/31 llb
# #        writer.add_scalars('train_loss', {'train_loss': sum(epoche_loss) / len(epoche_loss),
# #                                          'val_loss': sum(epoche_loss_val) / len(epoche_loss_val)}, epoch)
# #        writer.add_scalars('train_acc', {'train_acc': sum(epoche_acc) / len(epoche_acc),
# #                                         'val_acc': sum(epoche_acc_val) / len(epoche_acc_val)}, epoch)
# #        writer.add_scalars('auc', {'auc': sum(epoche_auc) / len(epoche_auc), }, epoch)
#         ##
#         '''2023/3/24 end'''
#         if (epoch+1)%2 == 0:
#             test_loss = []
#             test_acc = []
#             test_Auc = []
#             for kv in range(lay_num):
#                 test_loss_p = []
#                 test_acc_p = []
#                 test_Auc_p = []
#                 test_data_pair = sample_test_batch(idx_test[kv], ori_labels[kv], ori_adj[kv])
#                 num_batch = len(test_data_pair)
#                 for test_batch in test_data_pair:
#                     test_node, test_targets = test_batch
#                     test_loss_t = graphsage[kv].loss(test_node, test_targets)
#                     test_acc_t = graphsage[kv].acc(test_node, test_targets)
#                     test_Auc_t = graphsage[kv].Auc(test_node, test_targets)
#                     test_loss_p.append(test_loss_t)
#                     test_acc_p.append(test_acc_t)
#                     test_Auc_p.append(test_Auc_t)
#                 test_loss.append(sum(test_loss_p)/num_batch)
#                 test_acc.append(sum(test_acc_p)/num_batch)
#                 test_Auc.append(sum(test_Auc_p)/num_batch)
#             test_loss = (sum(test_loss)) / lay_num
#             test_acc = (sum(test_acc)) / lay_num
#             test_Auc = (sum(test_Auc)) / lay_num
#             print("test loss: {:.4f}  test acc: {:.4f}  test Auc: {:.4f}".format(test_loss, test_acc, test_Auc))
#
#             if test_Auc > best_test:
#                  best_epoch = epoch
#                  for k_i in range(lay_num):
#                      torch.save(graphsage[k_i].state_dict(), './new-ablation/celegans/semantic/_{}_{}.pkl'.format(best_epoch,k_i))
#                  best_test = test_Auc
#                  patience = 0
#             else:
#                  patience = patience + 1
#             if patience > 5 & epoch > 100:
#                  break
#             files = glob.glob('./new-ablation/celegans/semantic/*.pkl')
#             for file in files:
#                 epoch_nb = int(file.split('_')[1])
#                 if epoch_nb < (best_epoch-1):
#                     os.remove(file)
#     '''patience start'''
#     files = glob.glob('./new-ablation/celegans/semantic/*.pkl')
#     for file in files:
#         epoch_nb = int(file.split('_')[1])
#         if epoch_nb > best_epoch:
#             os.remove(file)
    for k_j in range(lay_num):
        graphsage[k_j].load_state_dict(torch.load('./new-ablation/ckm/semantic/_{}_{}.pkl'.format(307,k_j)))#best_epoch

    test_all = []
    all_layer_embeds=[];all_layer_predict=[];
    test_loss = []
    test_acc = []
    test_Auc = []
    for kv in range(lay_num):
        test_loss_p = []
        test_acc_p = []
        test_Auc_p = []
        test_data_pair = sample_test_batch(idx_test[kv], ori_labels[kv], ori_adj[kv])
        test_all.append(test_data_pair)
        num_batch = len(test_data_pair)
        layer_embeds=[];layer_predict=[];
        for test_batch in test_data_pair:
            test_node, test_targets = test_batch
            batch_embeds,batch_predict = graphsage[kv].get_data(test_node)
            layer_embeds.append(batch_embeds)
            layer_predict.append(batch_predict)
            test_loss_t = graphsage[kv].loss(test_node, test_targets)
            test_acc_t = graphsage[kv].acc(test_node, test_targets)
            test_Auc_t = graphsage[kv].Auc(test_node, test_targets)
            test_loss_p.append(test_loss_t)
            test_acc_p.append(test_acc_t)
            test_Auc_p.append(test_Auc_t)
        test_loss.append(sum(test_loss_p)/num_batch)
        test_acc.append(sum(test_acc_p)/num_batch)
        test_Auc.append(sum(test_Auc_p)/num_batch)
        all_layer_embeds.append(layer_embeds)
        all_layer_predict.append(layer_predict)
    test_loss = (sum(test_loss)) / lay_num
    test_acc = (sum(test_acc)) / lay_num
    test_Auc = (sum(test_Auc)) / lay_num
    print("test loss: {:.4f}  test acc: {:.4f}  test Auc: {:.4f}".format(test_loss, test_acc, test_Auc))
    data_test = dict()
    data_test['test_all'] = test_all
    data_test['all_layer_embeds'] = all_layer_embeds
    data_test['all_layer_predict'] = all_layer_predict
    with open("./new-ablation/ckm/semantic/ckm-second-semantic.pkl", "wb") as f:
        pickle.dump(data_test, f)
    f.close()
    sys.exit()

    # test_loss = []
    # test_acc = []
    # test_Auc = []
    # for kv in range(lay_num):
    #     test_loss_p = []
    #     test_acc_p = []
    #     test_Auc_p = []
    #     test_data_pair = sample_test_batch(idx_test[kv], ori_labels[kv], ori_adj[kv])
    #     num_batch = len(test_data_pair)
    #     for test_batch in test_data_pair:
    #         test_node, test_targets = test_batch
    #         test_loss_t = graphsage[kv].loss(test_node, test_targets)
    #         test_acc_t = graphsage[kv].acc(test_node, test_targets)
    #         test_Auc_t = graphsage[kv].Auc(test_node, test_targets)
    #         test_loss_p.append(test_loss_t)
    #         test_acc_p.append(test_acc_t)
    #         test_Auc_p.append(test_Auc_t)
    #     test_loss.append(sum(test_loss_p)/num_batch)
    #     test_acc.append(sum(test_acc_p)/num_batch)
    #     test_Auc.append(sum(test_Auc_p)/num_batch)
    # test_loss = (sum(test_loss)) / lay_num
    # test_acc = (sum(test_acc)) / lay_num
    # test_Auc = (sum(test_Auc)) / lay_num
    # print("test loss: {:.4f}  test acc: {:.4f}  test Auc: {:.4f}".format(test_loss, test_acc, test_Auc))
    # sys.exit()

    ## 2022/10/31 llb
#    writer.add_scalar('test_loss', test_loss, epoch)
#    writer.add_scalar('test_acc', test_acc, epoch)
#    writer.add_scalar('test_Auc', test_Auc, epoch)
    ##

    return test_acc, test_Auc, acc, auc, loss


if __name__ == "__main__":
    accs = [];
    aucs = [];
    losses = [];
    train_losses = [];
    train_accs = [];
    train_aucs = []
    # node_nums = [2708,3327,1436,19717]
    # graphs = ['cora','citeseer','vc','pubmed']

    node_nums = [246]
    graphs = ['ckm']


    # node_nums = [71]
    # graphs = ['lazega']

    # node_nums = [279]
    # graphs = ['celegans']

    # node_nums = [3025]
    # graphs = ['acm']

    #    node_nums = [3550]
    #    graphs = ['imdb']

#    node_nums = [7907]
#    graphs = ['dblp']
#
#    node_nums = [7621]
#    graphs = ['amazon']

    for cnt, dataname in enumerate(graphs):
        node_num = node_nums[cnt]
        emb_dim = [128]
#        lrs = [0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 0.7, 0.9]
        for dim in emb_dim:
            print('embedding dim {}'.format(dim))

            ## 2022/10/31 llb
#            writer = SummaryWriter(comment='node_num_{}_dataname_{}_dim_{}'.format(node_num, dataname, dim))
            ##

            Acc = []
            Auc = []
            Train_loss = []
            Train_acc = []
            Train_auc = []
            for times in range(1):
                acc, auc, train_acc, train_auc, train_loss = run(emb_dim=128, times=times, dataname=dataname,
                                                                 num_samples1=10, num_samples2=10, batch_num=256,
                                                                 break_por=0.05, epoches=400)
                Acc.append(acc.cpu().numpy())
                Auc.append(auc)
                # Loss.append(loss.cpu().detach().numpy())

                Train_acc.append(train_acc.cpu().numpy())
                Train_auc.append(train_auc)
                Train_loss.append(train_loss.cpu().detach().numpy())
            accs.append(np.array(Acc))
            # aucs.append(Auc)
            aucs.append(np.array(Auc))
            train_accs.append(np.array(Train_acc))
            train_aucs.append(np.array(Train_auc))
            train_losses.append(np.array(Train_loss))

            ## 2022/10/31 llb
            # export scalar data to JSON for external processing
#            writer.export_scalars_to_json("./all_scalars.json")
#            writer.close()
            ###

        # llb：原dataName改为dataname
        np.save('results/test4/{}_Acc.npy'.format(dataname), accs)
        np.save('results/test4/{}_Auc.npy'.format(dataname), aucs)
        np.save('results/test4/{}_TrianAcc.npy'.format(dataname), train_accs)
        np.save('results/test4/{}_TrianAuc.npy'.format(dataname), train_aucs)
        np.save('results/test4/{}_TrainLoss.npy'.format(dataname), train_losses)
