"""
@Time ： 2023/7/7 16:58
@Auth ： llb
@File ：test.py
@IDE ：PyCharm
"""
import sys
import matplotlib.pyplot as plt
from math import isclose
from sklearn.decomposition import PCA
import os
import networkx as nx
import numpy as np
import pandas as pd
from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter
from collections import Counter
import multiprocessing
# from IPython.display import display, HTML
from sklearn.model_selection import train_test_split
import pickle as pkl

'''其他数据集 start'''
#graph = nx.read_edgelist('./data/test.edgelist', nodetype=int, create_using=nx.Graph())
#graph = graph.to_undirected()
#
#graph = StellarGraph(graph)
#'''其他数据集 end'''
#
#link_data = pkl.load(open('data/amazon_link.pkl', "rb"))
#links = link_data['link']
#links = links[1]
'''end'''

'''cora\imdb数据集'''
graph = nx.read_edgelist('./data/cora.edgelist', nodetype=int, create_using=nx.Graph())
graph = graph.to_undirected()

graph = StellarGraph(graph)

community = []
file = open('./data/cora_louvain.txt','r')
for line in file.readlines():
    temp = line.strip().split('[')
    community.append([int(num) for num in temp[1].split(',')])
file.close()

non_label_non_comm = [] # 社区内不存在的连边
label_non_comm = [] # 社区内存在的连边
non_label_comm = []  # 社区间不存在的连边
label_comm = []  # 社区间存在的连边
data_adj = pkl.load(open('data/cora_adj.pkl', "rb"))
ori_adj = data_adj['adj'] # 获取第几层的社团内和社团间连边信息
for nl1 in range(len(community)):
    list_one = community[nl1]
    for nl2 in range(len(community)):
        list_two = community[nl2]
        temp_adj = ori_adj[list_one,:][:,list_two]
        for num_ns_1 in range(temp_adj.shape[0]):
            for num_ns_2 in range(temp_adj.shape[1]):
                if nl1 == nl2:  
#                        intra_link.append((list_one[num_ns_1],list_two[num_ns_2]))# intra link  包括存在的和不存在的
                     if temp_adj[num_ns_1][num_ns_2]==0:
                         non_label_non_comm.append((list_one[num_ns_1],list_two[num_ns_2]))
                     else:
                         label_non_comm.append((list_one[num_ns_1],list_two[num_ns_2]))
                else:
#                        inter_link.append((list_one[num_ns_1],list_two[num_ns_2]))# inter link  包括存在的和不存在的
                     if temp_adj[num_ns_1][num_ns_2]==0:
                         non_label_comm.append((list_one[num_ns_1],list_two[num_ns_2]))
                     else:
                         label_comm.append((list_one[num_ns_1],list_two[num_ns_2]))
links = [non_label_non_comm,label_non_comm,non_label_comm,label_comm]
'''end'''

#dataset = datasets.Cora()
#display(HTML(dataset.description))
#graph, _ = dataset.load(largest_connected_component_only=True, str_node_ids=True)

print('graph.info()')
print(graph.info())
print('----------------------------------------')

# Define an edge splitter on the original graph:
edge_splitter_test = EdgeSplitter(graph)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from graph, and obtain the
# reduced graph graph_test with the sampled links removed:
graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global"
)
print('graph_test.info()')
print(graph_test.info())
print('----------------------------------------')
# Do the same process to compute a training subset from within the test graph
edge_splitter_train = EdgeSplitter(graph_test, graph)
graph_train, examples, labels = edge_splitter_train.train_test_split(
    p=0.1, method="global"
)
(
    examples_train,
    examples_model_selection,
    labels_train,
    labels_model_selection,
) = train_test_split(examples, labels, train_size=0.75, test_size=0.25)
print('graph_train.info()')
print(graph_train.info())
print(type(graph_train))
print('----------------------------------------')

pd.DataFrame(
    [
        (
            "Training Set",
            len(examples_train),
            "Train Graph",
            "Test Graph",
            "Train the Link Classifier",
        ),
        (
            "Model Selection",
            len(examples_model_selection),
            "Train Graph",
            "Test Graph",
            "Select the best Link Classifier model",
        ),
        (
            "Test set",
            len(examples_test),
            "Test Graph",
            "Full Graph",
            "Evaluate the best Link Classifier",
        ),
    ],
    columns=("Split", "Number of Examples", "Hidden from", "Picked from", "Use"),
).set_index("Split")


p = 1.0
q = 1.0
dimensions = 128
num_walks = 10
walk_length = 80
window_size = 10
num_iter = 1
workers = multiprocessing.cpu_count()

from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec

def node2vec_embedding(graph, name):
    rw = BiasedRandomWalk(graph)
    walks = rw.run(graph.nodes(), n=num_walks, length=walk_length, p=p, q=q)
    print(f"Number of random walks for '{name}': {len(walks)}")

    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        workers=workers,
        epochs=num_iter
        )

    def get_embedding(u):
        return model.wv[u]

    return get_embedding


'''获取所有节点的嵌入'''
#embedding_all_node = node2vec_embedding(graph, "All nodes")
#sys.exit()
''''''


embedding_train = node2vec_embedding(graph_train, "Train Graph")

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# 1. link embeddings
def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [
        binary_operator(transform_node(src), transform_node(dst))
        for src, dst in link_examples
    ]


# 2. training classifier
def train_link_prediction_model(
    link_examples, link_labels, get_embedding, binary_operator
):
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator
    )
    clf.fit(link_features, link_labels)
    return clf


def link_prediction_classifier(max_iter=2000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])


# 3. and 4. evaluate classifier
def evaluate_link_prediction_model(
    clf, link_examples_test, link_labels_test, get_embedding, binary_operator,is_test
):
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator
    )
    score = evaluate_roc_auc(clf, link_features_test, link_labels_test,link_examples_test,is_test)
    return score


def evaluate_roc_auc(clf, link_features, link_labels,link_examples_test,is_test):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    prediction = predicted[:, positive_column]

    '''概率分布 start'''
    if is_test == 1:
        link_id = []
        num_link_comp = [[0 for _ in range(10)] for _ in range(4)]
        link_pre = [[] for _ in range(4)]
        for item in link_examples_test:
            if (item[0],item[1]) in links[0]:
                link_id.append(0)
            elif (item[0],item[1]) in links[1]:
                link_id.append(1)
            elif (item[0],item[1]) in links[2]:
                link_id.append(2)
            elif (item[0],item[1]) in links[3]:
                link_id.append(3)
            else:
                print("has one not belong any community list")
        num_inter_err = 0
        num_err = 0 
        for i in range(len(prediction)):
            if prediction[i] <= 0.5:
                num_err=num_err+1
                if link_id[i] == 1:
                    num_inter_err=num_inter_err+1
            '''计算各概率下各种类型连边的数量'''
            link_pre[link_id[i]].append(prediction[i])
            if prediction[i] >=0 and prediction[i] < 0.1:#概率为0-0.1
                num_link_comp[link_id[i]][0]+=1
            elif prediction[i] >=0.1 and prediction[i] < 0.2:#概率为0.1-0.2
                num_link_comp[link_id[i]][1]+=1
            elif prediction[i] >=0.2 and prediction[i] < 0.3:#概率为0.2-0.3
                num_link_comp[link_id[i]][2]+=1
            elif prediction[i] >=0.3 and prediction[i] < 0.4:#概率为0.3-0.4
                num_link_comp[link_id[i]][3]+=1
            elif prediction[i] >=0.4 and prediction[i] < 0.5:#概率为0.4-0.5
                num_link_comp[link_id[i]][4]+=1
            elif prediction[i] >=0.5 and prediction[i] < 0.6:#概率为0.5-0.6
                num_link_comp[link_id[i]][5]+=1
            elif prediction[i] >=0.6 and prediction[i] < 0.7:#概率为0.6-0.7
                num_link_comp[link_id[i]][6]+=1
            elif prediction[i] >=0.7 and prediction[i] < 0.8:#概率为0.7-0.8
                num_link_comp[link_id[i]][7]+=1
            elif prediction[i] >=0.8 and prediction[i] < 0.9:#概率为0.8-0.9
                num_link_comp[link_id[i]][8]+=1
            elif prediction[i] >=0.9 and prediction[i] <= 1:#概率为0.9-1
                num_link_comp[link_id[i]][9]+=1
            else:
                print('has error: this prediction value not belong section.')
                print(y_scores[i])
        if num_err==0:
            per_inter_err = 0
        else:
            per_inter_err = num_inter_err/num_err
        data_pre = dict()
        data_pre['prediction'] = link_pre
        data_pre['per_inter_err'] = per_inter_err
        with open("./cora_prediction.pkl", "wb") as f:
            pkl.dump(data_pre, f)
        f.close()
        print("----------------------------------")
        print(prediction)
        print(per_inter_err)
        print(num_link_comp)
        print("----------------------------------")
    '''概率分布 end'''
#    print(link_examples_test[0][0])
#    print(link_examples_test[0][1])
#    print(len(link_examples_test))
#    print(link_labels)
#    print(predicted[:, positive_column])
#    print(len(link_labels))
#    print(len(predicted[:, positive_column]))
    return roc_auc_score(link_labels, prediction)

def operator_hadamard(u, v):
    return u * v


def operator_l1(u, v):
    return np.abs(u - v)


def operator_l2(u, v):
    return (u - v) ** 2


def operator_avg(u, v):
    return (u + v) / 2.0


def run_link_prediction(binary_operator):
    clf = train_link_prediction_model(
        examples_train, labels_train, embedding_train, binary_operator
    )
    score = evaluate_link_prediction_model(
        clf,
        examples_model_selection,
        labels_model_selection,
        embedding_train,
        binary_operator,
        0,
    )

    return {
        "classifier": clf,
        "binary_operator": binary_operator,
        "score": score,
    }


binary_operators = [operator_hadamard, operator_l1, operator_l2, operator_avg]
results = [run_link_prediction(op) for op in binary_operators]
best_result = max(results, key=lambda result: result["score"])

print(f"Best result from '{best_result['binary_operator'].__name__}'")

pd.DataFrame(
    [(result["binary_operator"].__name__, result["score"]) for result in results],
    columns=("name", "ROC AUC score"),
).set_index("name")

embedding_test = node2vec_embedding(graph_test, "Test Graph")
test_score = evaluate_link_prediction_model(
    best_result["classifier"],
    examples_test,
    labels_test,
    embedding_test,
    best_result["binary_operator"],
    1,
)
print(
    f"ROC AUC score on test set using '{best_result['binary_operator'].__name__}': {test_score}"
)
