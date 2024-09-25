import numpy
import pickle as pkl
import sys
import torch
from sklearn.metrics import roc_auc_score, f1_score
if 1:
    data_dic = pkl.load(open('../community_res/metis/lazega-dic.pkl','rb'))
    data_dic = data_dic['dic_all']
    data = pkl.load(open('./single/lazega-single.pkl','rb'))
    print(data.keys())

    all_layer_nodes = [];all_layer_targets=[]
    for it in range(3):
        all_layer_nodes.append(data['test_all'][it][0][0])
        all_layer_targets.append(data['test_all'][it][0][1])
    print(len(all_layer_nodes))
    print(len(all_layer_targets))
    # all_layer_nodes = data['all_layer_nodes']
    # all_layer_targets = data['all_layer_targets']
    all_layer_predict = data['all_layer_predict']
    for layer in range(3):
        layer_dic = data_dic[layer]
        layer_node = [[int(item) for item in all_layer_nodes[layer]]]
        layer_target = [[int(item) for item in all_layer_targets[layer]]]
        layer_predict = all_layer_predict[layer]
        assert len(layer_target)==len(layer_predict)

        y_true = []
        y_pred = []
        y_pred_label = []

        for i in range(len(layer_target)):
            s_layer_node = layer_node[i]
            s_layer_target = layer_target[i]
            s_layer_predict = layer_predict[i]
            i_index = s_layer_node[::2]
            j_index = s_layer_node[1::2]
            assert len(i_index)==len(j_index) and len(i_index)==len(s_layer_target) and len(i_index)==len(s_layer_predict) 
            for j in range(len(i_index)):
                if layer_dic[int(i_index[j])] != layer_dic[int(j_index[j])]:
                    y_true.append(s_layer_target[j])
                    y_pred.append(s_layer_predict[j].item())
                    y_pred_label.append(1 if s_layer_predict[j]>=0.5 else 0 )
        assert len(y_true)==len(y_pred) and len(y_pred)==len(y_pred_label)
        print('len(y_true),len(y_pred),len(y_pred_label): {} , {} , {}'.format(len(y_true),len(y_pred),len(y_pred_label)))
        print('layer_num: {} | weak tie : auc: {}  |  f1: {}'.format(layer+1,roc_auc_score(y_true,y_pred),f1_score(y_true,y_pred_label)))
    print('--------------------------------')
