"""
@Time ： 2023/3/26 22:27
@Auth ： llb
@File ：Layweight.py
@IDE ：PyCharm
"""
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class NodeLayWeight(nn.Module):
    def __init__(self,embed_dim,features_nn):
        super(NodeLayWeight, self).__init__()
        self.features_nn = features_nn
        self.u = nn.Parameter(torch.FloatTensor((self.features_nn(torch.LongTensor([0]).cuda(0)).shape[1])*2,1))
        init.xavier_uniform_(self.u)
#        self.fea_=torch.tensor(fea_,dtype=torch.float).cuda()
#        self.trans_ = nn.Parameter(torch.FloatTensor(fea_.shape[1],embed_dim))
#        init.xavier_uniform_(self.trans_)

        self.LeakReLU = nn.LeakyReLU(0.1)

    def forward(self, node,to_neigh,num_sample):
#        print(self.features_nn(torch.LongTensor([93]).cuda()).shape)
#        print(node)
#        print(to_neigh)
#        print(self.features_nn(torch.LongTensor([node]).cuda()).shape)
#        print(self.features_nn(torch.LongTensor(list(to_neigh)).cuda()).shape)
#        print('============================')
        if type(node)==torch.Tensor:
            node = int(node.item())
        node_neigh = list(to_neigh)
        node_neigh_fea = self.features_nn(torch.LongTensor(list(to_neigh)))
        node_center_fea = self.features_nn(torch.LongTensor([node])).repeat(node_neigh_fea.shape[0],1)
        weight = torch.exp(self.LeakReLU(torch.matmul(torch.cat([node_neigh_fea,node_center_fea],1),self.u)))
        #weight = torch.exp(torch.matmul(torch.cat([node_neigh_fea,node_center_fea],1),self.u))
        weight = weight / sum(weight)
        selectV_I = torch.topk(weight, num_sample, 0, largest=True, sorted=False)
        neigh_index = selectV_I.indices.reshape(1,-1).squeeze().tolist()
        node_neigh = {node_neigh[i] for i in neigh_index}
        return node_neigh
        '''
        print(node_center_fea.shape)
        print(node_neigh_fea.shape)
        torch.Size([1, 128])
        torch.Size([8, 128])
        '''

class NodeLayWeightCos:
    def __init__(self,fea_):
        self.fea_ = torch.tensor(fea_)
        
    def computeCos(self,node,to_neigh,num_sample):
        node_neigh = list(to_neigh)
        node_neigh_fea = self.fea_[node_neigh]
        if type(node)==torch.Tensor:
            node = int(node.item())
        node_center_fea = self.fea_[node]
        weight = torch.zeros((node_neigh_fea.shape[0],1))
        for k_i in range(node_neigh_fea.shape[0]):
            weight[k_i]=torch.nn.functional.cosine_similarity(node_center_fea,node_neigh_fea[k_i],dim=0)
        selectV_I = torch.topk(weight, num_sample, 0, largest=True, sorted=False)
        neigh_index = selectV_I.indices.reshape(1,-1).squeeze().tolist()
        node_neigh = {node_neigh[i] for i in neigh_index}
        return node_neigh