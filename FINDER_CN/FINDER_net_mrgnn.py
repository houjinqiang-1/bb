from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_sparse
import numpy as np
from MRGNN.encoders import Encoder
from MRGNN.aggregators import MeanAggregator, LSTMAggregator, PoolAggregator
from MRGNN.utils import LogisticRegression
from MRGNN.mutil_layer_weight import LayerNodeAttention_weight, Cosine_similarity, SemanticAttention, \
    BitwiseMultipyLogis


# cudnn.benchmark = False
# cudnn.deterministic = True
# random.seed(0)
# np.random.seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# out = torch_sparse.spmm(index, value, m, n, matrix)
class FINDER_net(nn.Module):
    def __init__(self, SupervisedGraphSage, Encoder1, Encoder2, Aggregator, layerNodeAttention_weight,
                 embedding_size=64, w_initialization_std=1, reg_hidden=32, max_bp_iter=3,
                 embeddingMethod=1, aux_dim=4, device=None, node_attr=False):
        super(FINDER_net, self).__init__()

        self.SupervisedGraphSage = SupervisedGraphSage
        self.Encoder1 = Encoder1
        self.Encoder2 = Encoder2
        self.Aggregator = Aggregator
        self.layerNodeAttention_weight = layerNodeAttention_weight
        # self.rand_generator = torch.normal
        # see https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
        self.rand_generator = lambda mean, std, size: torch.fmod(torch.normal(mean, std, size=size), 2)
        self.embedding_size = embedding_size
        self.w_initialization_std = w_initialization_std
        self.reg_hidden = reg_hidden
        self.max_bp_iter = max_bp_iter
        self.embeddingMethod = embeddingMethod
        self.aux_dim = aux_dim
        self.device = device
        self.node_attr = node_attr
        self.act = nn.ReLU()
        
        # [2, embed_dim]
        self.w_n2l = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                     size=(2, self.embedding_size)))
        # [embed_dim, embed_dim]
        self.p_node_conv = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std,\
                                                                           size=(self.embedding_size, self.embedding_size)))
        # [reg_hidden+aux_dim, 1]
        if self.reg_hidden > 0:
            # [embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.truncated_normal([self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            self.h1_weight = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                             size=(
                                                                                 self.embedding_size, self.reg_hidden)))

            # [reg_hidden+aux_dim, 1]
            # h2_weight = tf.Variable(tf.truncated_normal([self.reg_hidden + aux_dim, 1], stddev=initialization_stddev), tf.float32)
            self.h2_weight = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                             size=(self.reg_hidden + self.aux_dim, 1)))
            # [reg_hidden2 + aux_dim, 1]
            self.last_w = self.h2_weight
        else:
            # [2*embed_dim, reg_hidden]
            # h1_weight = tf.Variable(tf.truncated_normal([2 * self.embedding_size, self.reg_hidden], stddev=initialization_stddev), tf.float32)
            self.h1_weight = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                             size=(
                                                                                 2 * self.embedding_size,
                                                                                 self.reg_hidden)))
            # [2*embed_dim, reg_hidden]
            self.last_w = self.h1_weight

        ## [embed_dim, 1]
        # cross_product = tf.Variable(tf.truncated_normal([self.embedding_size, 1], stddev=initialization_stddev), tf.float32)
        self.cross_product = nn.parameter.Parameter(data=self.rand_generator(0, self.w_initialization_std, \
                                                                             size=(self.embedding_size, 1)))

    def train_forward(self, node_input, subgsum_param, n2nsum_param, action_select, aux_input, adj, v_adj):
        
        nodes_cnt = n2nsum_param[0]['m']
        if (self.node_attr == False):
            # [node_cnt, 2]
            node_input = torch.ones((nodes_cnt, 2)).to(self.device)
                                 
        y_nodes_size = subgsum_param[0]['m']
        y_node_input = torch.ones((y_nodes_size,2)).to(self.device)
        
        input_message = torch.matmul(node_input, self.w_n2l)
        #[node_cnt, embed_dim]  # no sparse
        #input_potential_layer = tf.nn.relu(input_message)
        input_potential_layer = self.act(input_message)

        # # no sparse
        # [batch_size, embed_dim]
        #y_input_message = tf.matmul(tf.cast(y_node_input,tf.float32), w_n2l)
        y_input_message = torch.matmul(y_node_input, self.w_n2l)
        #[batch_size, embed_dim]  # no sparse
        #y_input_potential_layer = tf.nn.relu(y_input_message)
        y_input_potential_layer = self.act(y_input_message)

        #[node_cnt, embed_dim], no sparse
        cur_message_layer = input_potential_layer
        #cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
        cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)

        #[batch_size, embed_dim], no sparse
        y_cur_message_layer = y_input_potential_layer
        # [batch_size, embed_dim]
        #y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)
        y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)
        
        n2npool0 = torch_sparse.spmm(n2nsum_param[0]['index'], n2nsum_param[0]['value'],\
                n2nsum_param[0]['m'], n2nsum_param[0]['n'], cur_message_layer)
        #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
        #node_linear = tf.matmul(n2npool, p_node_conv)
        node_linear0 = torch.matmul(n2npool0, self.p_node_conv)

        n2npool1 = torch_sparse.spmm(n2nsum_param[1]['index'], n2nsum_param[1]['value'],\
                n2nsum_param[1]['m'], n2nsum_param[1]['n'], cur_message_layer)
        #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
        #node_linear = tf.matmul(n2npool, p_node_conv)
        node_linear1 = torch.matmul(n2npool1, self.p_node_conv)
        # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
        
        #OLD y_n2npool = torch.matmul(subgsum_param, cur_message_layer)
        y_n2npool0 = torch_sparse.spmm(subgsum_param[0]['index'], subgsum_param[0]['value'],\
                subgsum_param[0]['m'], subgsum_param[0]['n'], cur_message_layer)

        #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
        y_node_linear0 = torch.matmul(y_n2npool0, self.p_node_conv)
        
        y_n2npool1 = torch_sparse.spmm(subgsum_param[1]['index'], subgsum_param[1]['value'],\
                subgsum_param[1]['m'], subgsum_param[1]['n'], cur_message_layer)

        #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
        y_node_linear1 = torch.matmul(y_n2npool1, self.p_node_conv)
        
        node_input = torch.stack((node_linear0,node_linear1),axis=0)
        y_node_input = torch.stack((y_node_linear0,y_node_linear1),axis=0)
    
        node_input = torch.cat((node_input, y_node_input), axis=1)
        
        adj = np.array(adj)
        v_adj = np.array(v_adj)
        adj = np.concatenate((adj, np.zeros((adj.shape[0], adj.shape[1], y_nodes_size))), axis=2)
        v_adj = np.concatenate((v_adj, np.zeros((v_adj.shape[0], v_adj.shape[1], y_nodes_size))), axis=2)
        adj = np.concatenate((adj, v_adj), axis=1)
        if self.embeddingMethod == 1:  # MRGNN
            num_nodes = adj.shape[1]
            emb_dim = self.embedding_size
            adj_lists = []
            lay_num = 2
            for ki in range(lay_num):
                adj_lists_temp = defaultdict(set)
                [row, col] = np.where(adj[ki] == 1)
                for i in range(row.size):
                    adj_lists_temp[row[i]].add(col[i])
                    # adj_lists_temp[col[i]].add(row[i])
                adj_lists.append(adj_lists_temp)

            #features = []
            # for kik in range(lay_num):
            #     features_temp = nn.Embedding(num_nodes, node_input[kik].shape[1])
            #     features_temp.weight = nn.Parameter(torch.FloatTensor(node_input[kik].cpu()), requires_grad=True)
            #     features_temp.cuda(self.device)
            #     features.append(features_temp)
            nodes = np.array(list(range(nodes_cnt + y_nodes_size)))

            message_layer = self.SupervisedGraphSage(nodes, node_input, adj_lists, lay_num,
                                                     self.Encoder1, self.Encoder2, self.layerNodeAttention_weight,
                                                     self.Aggregator)
            cur_message_layer = message_layer[:, :nodes_cnt, :]
            y_cur_message_layer = message_layer[:, nodes_cnt:, :]
            cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=2)
            y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=2)
        q = 0
        for l in range(lay_num):
            node_embedding = cur_message_layer[l]
            # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim], dense
            y_potential = y_cur_message_layer[l].cuda(self.device)
            # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
            # action_embed = tf.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32), cur_message_layer)
            # OLD action_embed = torch.matmul(action_select, cur_message_layer)
            action_embed = torch_sparse.spmm(action_select[l]['index'], action_select[l]['value'], \
                                             action_select[l]['m'], action_select[l]['n'],
                                             cur_message_layer[l].cuda(self.device))

            # # [batch_size, embed_dim, embed_dim]
            # temp = tf.matmul(tf.expand_dims(action_embed, axis=2),tf.expand_dims(y_potential, axis=1))
            temp = torch.matmul(torch.unsqueeze(action_embed, dim=2), torch.unsqueeze(y_potential, dim=1))
            # [batch_size, embed_dim]
            # Shape = tf.shape(action_embed)
            Shape = action_embed.size()
            # [batch_size, embed_dim], first transform
            # embed_s_a = tf.reshape(tf.matmul(temp, tf.reshape(tf.tile(cross_product,[Shape[0],1]),[Shape[0],Shape[1],1])),Shape)
            embed_s_a = torch.reshape(torch.matmul(temp, torch.reshape(torch.tile(self.cross_product, [Shape[0], 1]), \
                                                                       [Shape[0], Shape[1], 1])), Shape)

            # [batch_size, 2 * embed_dim]
            last_output = embed_s_a
            if self.reg_hidden > 0:
                # [batch_size, 2*embed_dim] * [2*embed_dim, reg_hidden] = [batch_size, reg_hidden], dense
                # hidden = tf.matmul(embed_s_a, h1_weight)

                hidden = torch.matmul(embed_s_a, self.h1_weight)
                # [batch_size, reg_hidden]
                # last_output = tf.nn.relu(hidden)
                last_output = self.act(hidden)

            # if reg_hidden == 0: ,[[batch_size, 2*embed_dim], [batch_size, aux_dim]] = [batch_size, 2*embed_dim+aux_dim]
            # if reg_hidden > 0: ,[[batch_size, reg_hidden], [batch_size, aux_dim]] = [batch_size, reg_hidden+aux_dim]
            # last_output = tf.concat([last_output, self.aux_input], 1)
            last_output = torch.concat([last_output, aux_input[:,l,:]], 1)
            # if reg_hidden == 0: ,[batch_size, 2*embed_dim+aux_dim] * [2*embed_dim+aux_dim, 1] = [batch_size, 1]
            # if reg_hidden > 0: ,[batch_size, reg_hidden+aux_dim] * [reg_hidden+aux_dim, 1] = [batch_size, 1]
            # q_pred = tf.matmul(last_output, last_w)
            q_pred = torch.matmul(last_output, self.last_w)
            q += q_pred
        return q, cur_message_layer.cuda(self.device)

    def test_forward(self, node_input, subgsum_param, n2nsum_param, rep_global, aux_input, adj, v_adj):
        nodes_cnt = n2nsum_param[0]['m']
        if (self.node_attr == False):
            # [node_cnt, 2]
            node_input = torch.ones((nodes_cnt, 2)).to(self.device)
        y_nodes_size = subgsum_param[0]['m']
        y_node_input = torch.ones((y_nodes_size,2)).to(self.device)
        
        input_message = torch.matmul(node_input, self.w_n2l)
        #[node_cnt, embed_dim]  # no sparse
        #input_potential_layer = tf.nn.relu(input_message)
        input_potential_layer = self.act(input_message)

        # # no sparse
        # [batch_size, embed_dim]
        #y_input_message = tf.matmul(tf.cast(y_node_input,tf.float32), w_n2l)
        y_input_message = torch.matmul(y_node_input, self.w_n2l)
        #[batch_size, embed_dim]  # no sparse
        #y_input_potential_layer = tf.nn.relu(y_input_message)
        y_input_potential_layer = self.act(y_input_message)

        #[node_cnt, embed_dim], no sparse
        cur_message_layer = input_potential_layer
        #cur_message_layer = tf.nn.l2_normalize(cur_message_layer, axis=1)
        cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=1)

        #[batch_size, embed_dim], no sparse
        y_cur_message_layer = y_input_potential_layer
        # [batch_size, embed_dim]
        #y_cur_message_layer = tf.nn.l2_normalize(y_cur_message_layer, axis=1)
        y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=1)
        
        n2npool0 = torch_sparse.spmm(n2nsum_param[0]['index'], n2nsum_param[0]['value'],\
                n2nsum_param[0]['m'], n2nsum_param[0]['n'], cur_message_layer)
        #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
        #node_linear = tf.matmul(n2npool, p_node_conv)
        node_linear0 = torch.matmul(n2npool0, self.p_node_conv)

        n2npool1 = torch_sparse.spmm(n2nsum_param[1]['index'], n2nsum_param[1]['value'],\
                n2nsum_param[1]['m'], n2nsum_param[1]['n'], cur_message_layer)
        #[node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim], dense
        #node_linear = tf.matmul(n2npool, p_node_conv)
        node_linear1 = torch.matmul(n2npool1, self.p_node_conv)
        # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
        
        #OLD y_n2npool = torch.matmul(subgsum_param, cur_message_layer)
        y_n2npool0 = torch_sparse.spmm(subgsum_param[0]['index'], subgsum_param[0]['value'],\
                subgsum_param[0]['m'], subgsum_param[0]['n'], cur_message_layer)

        #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
        y_node_linear0 = torch.matmul(y_n2npool0, self.p_node_conv)
        
        y_n2npool1 = torch_sparse.spmm(subgsum_param[1]['index'], subgsum_param[1]['value'],\
                subgsum_param[1]['m'], subgsum_param[1]['n'], cur_message_layer)

        #[batch_size, embed_dim] * [embed_dim, embed_dim] = [batch_size, embed_dim], dense
        y_node_linear1 = torch.matmul(y_n2npool1, self.p_node_conv)
        
        node_input = torch.stack((node_linear0,node_linear1),axis=0)
        y_node_input = torch.stack((y_node_linear0,y_node_linear1),axis=0)
        node_input = torch.cat((node_input, y_node_input), axis=1)
        ######### 调试
        adj = np.array(adj)
        v_adj = np.array(v_adj)
        #########
        adj = np.concatenate((adj, np.zeros((adj.shape[0], adj.shape[1], y_nodes_size))), axis=2)
        v_adj = np.concatenate((v_adj, np.zeros((v_adj.shape[0], v_adj.shape[1], y_nodes_size))), axis=2)
        adj = np.concatenate((adj, v_adj), axis=1)
        if self.embeddingMethod == 1:  # MRGNN
            num_nodes = adj.shape[1]
            emb_dim = self.embedding_size
            adj_lists = []
            lay_num = 2
            for ki in range(lay_num):
                adj_lists_temp = defaultdict(set)
                [row, col] = np.where(adj[ki] == 1)
                for i in range(row.size):
                    adj_lists_temp[row[i]].add(col[i])
                    # adj_lists_temp[col[i]].add(row[i])
                adj_lists.append(adj_lists_temp)

            # features = []
            # for kik in range(lay_num):
            #     features_temp = nn.Embedding(num_nodes, node_input[kik].shape[1])
            #     features_temp.weight = nn.Parameter(torch.FloatTensor(node_input[kik].cpu()), requires_grad=True)
            #     features_temp.cuda(self.device)
            #     features.append(features_temp)

            nodes = np.array(list(range(nodes_cnt + y_nodes_size)))
            message_layer = self.SupervisedGraphSage(nodes, node_input, adj_lists, lay_num,
                                                     self.Encoder1, self.Encoder2, self.layerNodeAttention_weight,
                                                     self.Aggregator)
            cur_message_layer = message_layer[:, :nodes_cnt, :]
            y_cur_message_layer = message_layer[:, nodes_cnt:, :]
            cur_message_layer = torch.nn.functional.normalize(cur_message_layer, p=2, dim=2)
            y_cur_message_layer = torch.nn.functional.normalize(y_cur_message_layer, p=2, dim=2)

        q = 0
        for l in range(lay_num):
            y_potential = y_cur_message_layer[l].cuda(self.device)
            # [node_cnt, batch_size] * [batch_size, embed_dim] = [node_cnt, embed_dim]
            # OLD rep_y = torch.matmul(rep_global, y_potential)
            rep_y = torch_sparse.spmm(rep_global[l]['index'], rep_global[l]['value'].cuda(self.device), \
                                      rep_global[l]['m'], rep_global[l]['n'], y_potential.cuda(self.device))
            # [[node_cnt, embed_dim], [node_cnt, embed_dim]] = [node_cnt, 2*embed_dim]
            # embed_s_a_all = tf.concat([cur_message_layer,rep_y],1)
            # # [node_cnt, embed_dim, embed_dim]
            temp1 = torch.matmul(torch.unsqueeze(cur_message_layer[l].cuda(self.device), dim=2),
                                 torch.unsqueeze(rep_y, dim=1))
            # [node_cnt embed_dim]
            Shape1 = cur_message_layer[l].size()
            # [batch_size, embed_dim], first transform
            embed_s_a_all = torch.reshape(torch.matmul(temp1,
                                                       torch.reshape(torch.tile(self.cross_product, [Shape1[0], 1]),
                                                                     [Shape1[0], Shape1[1], 1])), Shape1)

            # [node_cnt, 2 * embed_dim]
            last_output = embed_s_a_all
            if self.reg_hidden > 0:
                # [node_cnt, 2 * embed_dim] * [2 * embed_dim, reg_hidden] = [node_cnt, reg_hidden1]
                hidden = torch.matmul(embed_s_a_all, self.h1_weight)
                # Relu, [node_cnt, reg_hidden1]
                last_output = self.act(hidden)
                # [node_cnt, reg_hidden1] * [reg_hidden1, reg_hidden2] = [node_cnt, reg_hidden2]
                # last_output_hidden = tf.matmul(last_output1, h2_weight)
                # last_output = tf.nn.relu(last_output_hidden)

            # [node_cnt, batch_size] * [batch_size, aux_dim] = [node_cnt, aux_dim]
            rep_aux = torch_sparse.spmm(rep_global[l]['index'], rep_global[l]['value'],\
                rep_global[l]['m'], rep_global[l]['n'], aux_input[:,l,:])
            # rep_aux = torch.matmul(rep_global, aux_input)

            # if reg_hidden == 0: , [[node_cnt, 2 * embed_dim], [node_cnt, aux_dim]] = [node_cnt, 2*embed_dim + aux_dim]
            # if reg_hidden > 0: , [[node_cnt, reg_hidden], [node_cnt, aux_dim]] = [node_cnt, reg_hidden + aux_dim]
            last_output = torch.concat([last_output, rep_aux], 1)

            # if reg_hidden == 0: , [node_cnt, 2 * embed_dim + aux_dim] * [2 * embed_dim + aux_dim, 1] = [node_cnt，1]
            # f reg_hidden > 0: , [node_cnt, reg_hidden + aux_dim] * [reg_hidden + aux_dim, 1] = [node_cnt，1]
            q_on_all = torch.matmul(last_output, self.last_w)
            q += q_on_all
        return q
