import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

'''
层间节点权重计算
'''
class LayerNodeAttention_weight(nn.Module):
    def __init__(self, features_num, dropout, alpha, metapath_number ,layer_predict=0):
        super(LayerNodeAttention_weight, self).__init__()
        self.features_num = features_num
        self.layer_predict = layer_predict
        self.metapath_number = metapath_number

        self.dropout = dropout
        self.alpha = alpha

        self.leakyReLU = nn.LeakyReLU(alpha)  # LeakyReLU激活函数

#        self.trans = nn.Parameter(torch.empty(features_num, features_num))
#        nn.init.xavier_uniform_(self.trans.data, 1.414)

        '''生成对角线矩阵'''
#        tempOne = nn.Parameter(torch.empty(1, features_num))
#        nn.init.xavier_uniform_(tempOne.data, 1.414)
#        tempTwo = torch.diag_embed(tempOne)
#        self.trans = tempTwo[0]

#        tempOne = torch.empty(1, features_num)
#        nn.init.xavier_uniform_(tempOne, 1.414)
        self.trans = nn.Parameter(torch.eye(features_num))
        '''end'''

        self.attention = nn.Parameter(torch.empty(1, 2 * features_num))
        nn.init.xavier_uniform_(self.attention.data, 1.414)

        self.bias = nn.Parameter(torch.Tensor(features_num))
        nn.init.zeros_(self.bias)

        self.tanh = nn.Tanh()



    def forward(self, node_features, nodes_ori):
        nodes_ori = np.array(nodes_ori)
        nodes_ori = nodes_ori.tolist()
        nodes = torch.tensor(np.unique(nodes_ori))
        temp_fea_t = []
        for kk in range(self.metapath_number):
            temp_fea_t.append(torch.zeros(len(nodes),node_features[kk].shape[1]))
        for i in range(len(nodes)):
            index_ = nodes_ori.index(nodes[i])
            for kj in range(self.metapath_number):
                temp_fea_t[kj][i] = node_features[kj][index_]
        for kl in range(self.metapath_number):
            node_features[kl] = torch.tanh(torch.matmul(temp_fea_t[kl],self.trans)+self.bias)
            # 转换为同一特征空间


        node_features = torch.stack(node_features)

        layer_all_attention = torch.stack([
            self.layer_node_attention(node_features, i) for i in range(node_features.shape[1])
        ]) # 得到层间每个节点的特征矩阵，Tensor=（node_number, meta_path, out_feaures）

        Z = torch.zeros(node_features.shape[1],node_features.shape[2])
        for i in range(node_features.shape[1]): # 循环每一个节点
            adj = layer_all_attention[i] # 第i个节点的层间特征矩阵 tensor=（meta_path, out_feaures）
            weight = [0 for i in range(self.metapath_number)]
            for j in range(adj.shape[0]):
                if j == self.layer_predict:
                    continue
                cat_hi = torch.cat((adj[self.layer_predict],adj[j]), dim=0)
                weight_t = math.exp(self.leakyReLU(self.attention.matmul(cat_hi)))
                weight[j] = weight_t if weight_t<1 else 1
            #print(weight)
            temp = Z[i]
            for k in range(adj.shape[0]):
                if k==self.layer_predict:
                    continue
                temp +=(weight[k] / sum(weight)) *adj[k]
                #print(weight[k] / sum(weight))
            Z[i] = temp
            #Z[i]=torch.sigmoid(Z[i])

        X = node_features[self.layer_predict]+Z
        result = torch.zeros(len(nodes_ori),X.shape[1])
        nodes_tolist = nodes.tolist()
        for m in range(len(nodes_ori)):
            index_nodes = nodes_tolist.index(nodes_ori[m])
            result[m] = X[index_nodes]
        return  result

    def transZshape(self, z, dim, i):
        # matrics为单位列矩阵，获取第几层节点的层间特征矩阵，单位矩阵第几行就为1
        matrics = torch.zeros(dim, 1)
        matrics[i, 0] = 1
        m = z.matmul(matrics).reshape(z.shape[0], z.shape[1])  # m=Tensor(node_number,out_features)
        return m # m为第i层节点的层间特征矩阵

    '''计算层间节点的特征矩阵
        node_features为层内节点注意力后的特征矩阵，三维
        i为第i个节点
        '''
    def layer_node_attention(self, node_features, i):
        a_temp = torch.zeros(node_features.shape[1], 1)  # 要得到i节点的层间特征矩阵就设第几行为1，其余为0
        a_temp[i, 0] = 1
        layer_attention = torch.transpose(node_features, 2, 1)
        b = layer_attention.matmul(a_temp).reshape(layer_attention.shape[0], layer_attention.shape[1])
        # b为第i个节点的层间特征矩阵 = Tensor(meta_path, out_features)
        return b


'''
余弦相似性
'''
class Cosine_similarity(nn.Module):
    def __init__(self, features_num, dropout, alpha, metapath_number, layer_predict=0):
        super(Cosine_similarity, self).__init__()
        self.features_num = features_num
        self.layer_predict = layer_predict
        self.metapath_number = metapath_number

        self.dropout = dropout
        self.alpha = alpha

        self.leakyReLU = nn.LeakyReLU(alpha)  # LeakyReLU激活函数

#        self.trans = nn.Parameter(torch.empty(features_num, features_num))
#        nn.init.xavier_uniform_(self.trans.data, 1.414)

        '''生成对角线矩阵'''
#        tempOne = nn.Parameter(torch.empty(1, features_num))
#        nn.init.xavier_uniform_(tempOne.data, 1.414)
#        tempTwo = torch.diag_embed(tempOne)
#        self.trans = tempTwo[0]
        self.trans = nn.Parameter(torch.eye(features_num))
        '''end'''

        self.attention = nn.Parameter(torch.empty(1, 2 * features_num))
        nn.init.xavier_uniform_(self.attention.data, 1.414)

        self.bias = nn.Parameter(torch.Tensor(features_num))
        nn.init.zeros_(self.bias)

    def forward(self, node_features1, node_features2, node_features3, nodes_ori):
        nodes_ori = np.array(nodes_ori)
        nodes_ori = nodes_ori.tolist()
        nodes = torch.tensor(np.unique(nodes_ori))
        temp_fea_1 = torch.zeros(len(nodes), node_features1.shape[1])
        temp_fea_2 = torch.zeros(len(nodes), node_features2.shape[1])
        temp_fea_3 = torch.zeros(len(nodes), node_features3.shape[1])
        for i in range(len(nodes)):
            index_ = nodes_ori.index(nodes[i])
            temp_fea_1[i] = node_features1[index_]
            temp_fea_2[i] = node_features2[index_]
            temp_fea_3[i] = node_features3[index_]
        node_features1 = temp_fea_1
        node_features2 = temp_fea_2
        node_features3 = temp_fea_3

        # 转换为同一特征空间
        node_features1 = torch.tanh(torch.matmul(node_features1,self.trans)+self.bias)
        node_features2 = torch.tanh(torch.matmul(node_features2,self.trans)+self.bias)
        node_features3 = torch.tanh(torch.matmul(node_features3,self.trans)+self.bias)

        node_features = torch.stack([node_features1, node_features2, node_features3])  # 层间特征矩阵拼接，变成三维

        layer_all_attention = torch.stack([
            self.layer_node_attention(node_features, i) for i in range(node_features.shape[1])
        ])  # 得到层间每个节点的特征矩阵，Tensor=（node_number, meta_path, out_feaures）

        Z = torch.zeros(node_features.shape[1],node_features.shape[2])
        for i in range(layer_all_attention.shape[0]):  # 循环每一个节点
            adj = layer_all_attention[i]  # 第i个节点的层间特征矩阵 tensor=（meta_path, out_feaures）

            weight = [0 for i in range(self.metapath_number)]
            for j in range(adj.shape[0]):
                if j == self.layer_predict:
                    continue
                weight[j] = F.cosine_similarity(adj[self.layer_predict], adj[j], dim=0)

            #weight = F.softmax(torch.tensor(weight), dim=0) #权重按行归一化
            temp = Z[i]
            for k in range(adj.shape[0]):
                if k == self.layer_predict:
                    continue
                temp = temp + (weight[k] / sum(weight)) * adj[k]
            Z[i] = temp


        X = node_features[self.layer_predict]+Z
        result = torch.zeros(len(nodes_ori), X.shape[1])
        nodes_tolist = nodes.tolist()
        for m in range(len(nodes_ori)):
            index_nodes = nodes_tolist.index(nodes_ori[m])
            result[m] = X[index_nodes]
        return result


    def layer_node_attention(self, node_features, i):
        a_temp = torch.zeros(node_features.shape[1], 1)  # 要得到i节点的层间特征矩阵就设第几行为1，其余为0
        a_temp[i, 0] = 1
        layer_attention = torch.transpose(node_features, 2, 1)
        b = layer_attention.matmul(a_temp).reshape(layer_attention.shape[0], layer_attention.shape[1])
        # b为第i个节点的层间特征矩阵 = Tensor(meta_path, out_features)
        return b

'''
跨层语义融合机制
'''
class SemanticAttention(nn.Module):
    def __init__(self, features_num, dropout, alpha, metapath_number ,layer_predict=0):
        super(SemanticAttention, self).__init__()
        self.features_num = features_num
        self.layer_predict = layer_predict
        self.metapath_number = metapath_number

        self.dropout = dropout
        self.alpha = alpha

        self.leakyReLU = nn.LeakyReLU(alpha)  # LeakyReLU激活函数

#        self.trans = nn.Parameter(torch.empty(features_num, features_num))
#        nn.init.xavier_uniform_(self.trans.data, 1.414)

        '''生成对角线矩阵'''
#        tempOne = nn.Parameter(torch.empty(1, features_num))
#        nn.init.xavier_uniform_(tempOne.data, 1.414)
#        tempTwo = torch.diag_embed(tempOne)
#        self.trans = tempTwo[0]
        self.trans = nn.Parameter(torch.eye(features_num))
        '''end'''

        self.attention = nn.Parameter(torch.empty(1, 2 * features_num))
        nn.init.xavier_uniform_(self.attention.data, 1.414)

        self.W = nn.Parameter(torch.empty(features_num, features_num))
        nn.init.xavier_uniform_(self.W.data, 1.414)

        self.b = nn.Parameter(torch.empty(1, features_num))
        nn.init.xavier_uniform_(self.b.data, 1.414)

        self.q = nn.Parameter(torch.empty(features_num, 1))
        nn.init.xavier_uniform_(self.q.data, 1.414)

        self.bias = nn.Parameter(torch.Tensor(features_num))
        nn.init.zeros_(self.bias)

        self.tanh = nn.Tanh()


    def forward(self, node_features, nodes_ori):
        nodes_ori = np.array(nodes_ori)
        nodes_ori = nodes_ori.tolist()
        nodes = torch.tensor(np.unique(nodes_ori))
        temp_fea_t = []
        for kk in range(self.metapath_number):
            temp_fea_t.append(torch.zeros(len(nodes),node_features[kk].shape[1]))
        for i in range(len(nodes)):
            index_ = nodes_ori.index(nodes[i])
            for kj in range(self.metapath_number):
                temp_fea_t[kj][i] = node_features[kj][index_]
        for kl in range(self.metapath_number):
            node_features[kl] = torch.tanh(torch.matmul(temp_fea_t[kl],self.trans)+self.bias)
            # 转换为同一特征空间
 
        node_features = torch.stack(node_features)

        layer_all_attention = torch.stack([
            self.layer_node_attention(node_features, i) for i in range(node_features.shape[1])
        ]) # 得到层间每个节点的特征矩阵，Tensor=（node_number, meta_path, out_feaures）
        semantic_pernode = self.layer_semantic(layer_all_attention)#Tensor(node_number,meta_path-1,2,out_feaures)
        Z = torch.zeros(node_features.shape[1],node_features.shape[2])
        all_weight = []###################################################################
        for i in range(semantic_pernode.shape[0]): # 循环每一个节点
            adj_node = layer_all_attention[i]#第i个节点的层间特征矩阵 Tensor=(meta_path,out_features)
            adj = semantic_pernode[i] #  tensor=（meta_path-1,2,out_feaures）
            trans = torch.tanh(adj.matmul(self.W)+self.b)
            w_meta = trans.matmul(self.q).reshape(trans.shape[0],trans.shape[1])#Tensor(meta_path-1,2)
            w_meta = w_meta.sum(dim=1) / w_meta.shape[1]#得到每条路径权重
            beta = F.softmax(w_meta,dim=-1)
#            print(beta)
#            sys.exit()

            weight = []
            for kk, weight_k in enumerate(beta):
                weight.append(weight_k)
            #print(weight)
            temp_adj = Z[i]
            index = 0
            for k_ in range(adj_node.shape[0]):
                if k_ == self.layer_predict:
                    continue
                temp_adj = temp_adj + (weight[index]/(sum(weight))) * adj_node[k_]
                index = index+1
            Z[i] = temp_adj

            all_weight.append([vitem.item() for vitem in weight])
            
        X = node_features[self.layer_predict]+Z
        result = torch.zeros(len(nodes_ori),X.shape[1])
        nodes_tolist = nodes.tolist()
        for m in range(len(nodes_ori)):
            index_nodes = nodes_tolist.index(nodes_ori[m])
            result[m] = X[index_nodes]
        return  result,all_weight


    def layer_node_attention(self, node_features, i):
        a_temp = torch.zeros(node_features.shape[1], 1)  # 要得到i节点的层间特征矩阵就设第几行为1，其余为0
        a_temp[i, 0] = 1
        layer_attention = torch.transpose(node_features, 2, 1)
        b = layer_attention.matmul(a_temp).reshape(layer_attention.shape[0], layer_attention.shape[1])
        # b为第i个节点的层间特征矩阵 = Tensor(meta_path, out_features)
        return b
    def layer_semantic(self,node_layer_feature):
        layer_semantic_ = torch.zeros(node_layer_feature.shape[0],self.metapath_number-1,2,self.features_num)
        for k in range(node_layer_feature.shape[0]):
            adj_pernode = node_layer_feature[k]
            temp_node = torch.zeros(self.metapath_number,2,self.features_num)
            temp_path = torch.zeros(2,self.features_num)
            temp_path[0] = adj_pernode[self.layer_predict]
            for j in range(self.metapath_number):
                if j==self.layer_predict:
                    continue
                temp_path[1] = adj_pernode[j]
                temp_node[j] = temp_path
            temp_node = temp_node[torch.arange(temp_node.size(0))!=self.layer_predict]
            layer_semantic_[k] =  temp_node
        return layer_semantic_

'''按位相乘逻辑回归'''
class BitwiseMultipyLogis(nn.Module):
    def __init__(self, features_num, dropout, alpha, metapath_number,device):
        super(BitwiseMultipyLogis, self).__init__()
        self.features_num = features_num
        self.metapath_number = metapath_number

        self.dropout = dropout
        self.alpha = alpha

        self.leakyReLU = nn.LeakyReLU(alpha)  # LeakyReLU激活函数
        self.logis = LogisticVector(features_num,1)
        self.device = device
#        self.trans = nn.Parameter(torch.empty(features_num, features_num))
#        nn.init.xavier_uniform_(self.trans.data, 1.414)

        '''生成对角线矩阵'''
        #        tempOne = nn.Parameter(torch.empty(1, features_num))
        #        nn.init.xavier_uniform_(tempOne.data, 1.414)
        #        tempTwo = torch.diag_embed(tempOne)
        #        self.trans = tempTwo[0]
        self.trans = nn.Parameter(torch.eye(features_num))
        '''end'''
        self.bias = nn.Parameter(torch.Tensor(features_num))
        nn.init.zeros_(self.bias)

    def forward(self, node_features, nodes_ori, layer_predict):
        nodes_ori = np.array(nodes_ori)
        nodes_ori = nodes_ori.tolist()
        nodes = torch.tensor(np.unique(nodes_ori))
        # temp_fea_t = []
        # for kk in range(self.metapath_number):
        #     temp_fea_t.append(torch.zeros(len(nodes),node_features[kk].shape[1]))
        # for i in range(len(nodes)):
        #     index_ = nodes_ori.index(nodes[i])
        #     for kj in range(self.metapath_number):
        #         temp_fea_t[kj][i] = node_features[kj][index_]
        node_features_temp = torch.zeros((2,node_features[0].size(0),node_features[0].size(1))).to(self.device)
        for kl in range(self.metapath_number):
            node_features_temp[kl] = torch.tanh(torch.matmul(node_features[kl],self.trans)+self.bias)
            # 转换为同一特征空间
        node_features = node_features_temp
        
        # layer_all_attention = torch.stack([
        #     self.layer_node_attention(node_features, i) for i in range(node_features.shape[1])
#        ])  
        layer_all_attention = torch.transpose(node_features, 0, 1) # 得到层间每个节点的特征矩阵，Tensor=（node_number, meta_path, out_feaures）
        semantic_pernode = self.layer_bitwise(layer_all_attention,layer_predict)  # Tensor(node_number,meta_path,out_feaures)
        Z = torch.zeros(node_features.shape[1], node_features.shape[2]).cuda(self.device)
        all_weight = []###################################################################
#         for i in range(semantic_pernode.shape[0]):  # 循环每一个节点
#             adj_node = layer_all_attention[i]  # 第i个节点的层间特征矩阵 Tensor=(meta_path,out_features)
#             adj = semantic_pernode[i]  # tensor=（meta_path,out_feaures）,此处已完成预测层节点向量分别与其他层节点向量按位相乘
#             weight = self.logis(adj)# weight未归一化，若归一化还得剔除预测层的权重再归一化
#             weight = F.softmax(weight, dim=0)
#             temp = Z[i]
#             wi = 0
# #            print(weight)
#             for kk in range(self.metapath_number):
#                 if kk == layer_predict:
#                     continue
#                 wi = kk
#                 temp = temp + weight[wi] * adj_node[kk]
#                 #wi = wi+1
#             Z[i] = temp
#             all_weight.append([vitem.item() for vitem in weight])
        adj_node = layer_all_attention  # 第i个节点的层间特征矩阵 Tensor=(meta_path,out_features)
        weight = self.logis(semantic_pernode)
        weight = F.softmax(weight, dim=1)
        for kk in range(self.metapath_number):
            if kk == layer_predict:
                continue
            Z = Z + weight[:,kk].unsqueeze(1) * adj_node[:,kk]
        X = node_features[layer_predict] + Z
        return X

    def layer_node_attention(self, node_features, i):
        a_temp = torch.zeros(node_features.shape[1], 1).cuda(self.device)  # 要得到i节点的层间特征矩阵就设第几行为1，其余为0
        a_temp[i, 0] = 1
        layer_attention = torch.transpose(node_features, 2, 1)
        b = layer_attention.matmul(a_temp).reshape(layer_attention.shape[0], layer_attention.shape[1])
        # b为第i个节点的层间特征矩阵 = Tensor(meta_path, out_features)
        return b

    def layer_bitwise(self, node_layer_feature,layer_predict):
        # layer_semantic = torch.zeros(node_layer_feature.shape[0], self.metapath_number, self.features_num).cuda(0)
        # for k in range(node_layer_feature.shape[0]):
        #     adj_pernode = node_layer_feature[k]
        #     temp_node = torch.zeros(self.metapath_number, self.features_num)
        #     for j in range(self.metapath_number):
        #         if j == layer_predict:
        #             continue
        #         temp_node[j] = torch.mul(adj_pernode[layer_predict],adj_pernode[j])
        #     #temp_node = temp_node[torch.arange(temp_node.size(0)) != self.layer_predict]
        #     layer_semantic[k] = temp_node
        
        # 创建一个全零张量来存储结果
        layer_semantic = torch.zeros_like(node_layer_feature).cuda(self.device)
        
        # 获取需要排除的维度的索引列表
        exclude_dims = [i for i in range(self.metapath_number) if i != layer_predict]
        
        # 对需要排除的维度进行广播和逐元素乘法
        excluded_features = node_layer_feature[:, exclude_dims] * node_layer_feature[:, layer_predict].unsqueeze(1)
        self_features =  node_layer_feature[:,layer_predict] * node_layer_feature[:, layer_predict]
        # 将结果存储到对应的位置
        layer_semantic[:, exclude_dims] = excluded_features
        layer_semantic[:, layer_predict] = self_features

        return layer_semantic


class LogisticVector(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(LogisticVector, self).__init__()
        self.n_feature = n_feature
        self.parameter = torch.nn.Linear(n_feature, n_hidden) # hidden layer
        self.active = nn.Sigmoid() ####  # output layer
    def forward(self,x):
        value = self.parameter(x)
        out = self.active(value)
        return out.squeeze()
class LogisticVector2(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(LogisticVector2, self).__init__()
        self.n_feature = n_feature
        self.parameter = torch.nn.Linear(n_feature, n_hidden) # hidden layer
        self.active = nn.Sigmoid() ####  # output layer
    def forward(self,x):
        value = self.parameter(x)
        out = self.active(value )
        return out.squeeze()