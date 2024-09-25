#coding=utf-8
import networkx as nx
import math
import random
from random import shuffle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

def add_edges_from_file(graph, file):
    for line in file:
        edge = line.strip().split(' '.encode())
        u, v = int(edge[0]), int(edge[1])
        graph.add_edge(u, v)
    return graph
def find_connected_components(graph):
    connected_components = []
    # print(graph.nodes)
    # print(type(graph.nodes))
    for node in graph.nodes:
        connected_component = nx.node_connected_component(graph, node)
        if connected_component not in connected_components:
            connected_components.append(connected_component)
    return connected_components


def find_integer_in_sets(integer, set_list):
    for index, integer_set in enumerate(set_list):
        if integer in integer_set:
            return index  # 返回整数所在的集合索引

def deledge(graph, connected_components):
    for (u, v) in graph.edges:
        # print(find_integer_in_sets(u, connected_components1))
        #如果u,v不在同一个联通分量的话就删除u,v的连边
        if v not in connected_components[find_integer_in_sets(u, connected_components)]:
            graph.remove_edge(u, v)

def find_max_set_length(set_list):
    max_length = 0
    for integer_set in set_list:
        set_length = len(integer_set)
        if set_length > max_length:
            max_length = set_length

    return max_length

def MCC(G1, G2):
    connected_components1 = find_connected_components(G1)
    connected_components2 = find_connected_components(G2)
    while connected_components1 != connected_components2:
        deledge(G2, connected_components1)
        connected_components2 = find_connected_components(G2)
        deledge(G1, connected_components2)
        connected_components1 = find_connected_components(G1)
    return connected_components1
def delnode(G1,G2):
    D1 = dict(G1.degree())
    D2 = dict(G2.degree())

    max_degrees = []
    for node in D1.keys():
        degree1 = D1[node]
        degree2 = D2[node]
        max_degrees.append((node, max(degree1, degree2)))

    # print(max_degrees)
    # D=max(max_degrees, key=lambda t:t[1])
    max_degree = max(max_degrees, key=lambda t: t[1])[1]
    max_degree_nodes = [t[0] for t in max_degrees if t[1] == max_degree]
    # print(len(max_degree_nodes))
    random_node = random.choice(max_degree_nodes)
    G1.remove_node(random_node)
    G2.remove_node(random_node)
    dN = 1

    return G1, G2, dN

def critical_number(G1, G2, N, M):
    MCCs = [1]
    ps = [1]
    dN = 0
    num = N
    G3 = G1.copy()
    G4 = G2.copy()
    lastm = M
    while num > 0:
        G5, G6, add_dN = delnode(G3, G4)
        num -= 1
        m = MCC(G5, G6)
        G3 = G5.copy()
        G4 = G6.copy()
        value = m / M
        if m <= 0.4 * M and m > math.sqrt(M):
            # if m < lastm:
            dN += add_dN
                # print('dN = {}    m/M = {}'.format(dN, value))
        # elif m <= math.sqrt(M):
            # if m < lastm:
            # dN += add_dN
                # print('dN = {}    m/M = {}'.format(dN, value))
            # break
        MCCs.append(value)
        ps.append(num/N)
        lastm = m
    dN += 1

    return dN, MCCs, ps

def draw_percolation_transition(fig, dataname, MCCs, ps, num1, num2, type, label):

    # 绘制MCC和p的关系图，线型为绿色实线
    plt.plot(ps, MCCs, type, label=label)

    # 添加标题和标签
    plt.title('{}_{}_{}'.format(dataname[:3], num1, num2))
    plt.xlabel('p')
    plt.ylabel('MCC')

    # 设置x轴和y轴的范围
    plt.xlim(0.7, 1.0)

    # 设置x轴的刻度间隔
    x_major_locator = MultipleLocator(0.1)  # 设置主刻度为0.5
    x_minor_locator = MultipleLocator(0.05)  # 设置次刻度为0.25

    # 将刻度设置应用于x轴
    plt.gca().xaxis.set_major_locator(x_major_locator)
    plt.gca().xaxis.set_minor_locator(x_minor_locator)

    return fig

def random_swap_elements(arr):
    n = len(arr) - 1  # 69  246

    n = n - n % 2  # 68  246

    # 创建一个用于跟踪元素是否已经被交换的列表
    swapped = [False] * n  # [0,n-1]

    for i in range(1, n+1):  # [1,n]
        if not swapped[i-1]:
            # 随机选择另一个尚未被交换的元素的索引
            j = random.randint(1, n)  # [1,n]

            # 确保不选择自己
            while j == i or swapped[j-1]:
                j = random.randint(1, n)

            # 交换元素
            arr[i], arr[j] = arr[j], arr[i]

            # 更新交换状态
            swapped[i-1] = swapped[j-1] = True

    return arr

def pre_data(dataname, num1, num2, N, reshuffle_layer):
    edgefile = datapath + 'multiplex_edges/' + dataname + '.edges'
    datafolder = datapath + dataname
    if not os.path.exists(datafolder):
        os.mkdir(datafolder)
    datafile = datafolder + '/data{}{}.txt'.format(num1, num2)
    data_r_file = datafolder + '/data_r{}{}.txt'.format(num1, num2)
    data = open(datafile, "w")
    d1 = open(datafolder + '/data{}{}_{}.txt'.format(num1, num2, num1), "w")
    d2 = open(datafolder + '/data{}{}_{}.txt'.format(num1, num2, num2), "w")
    data_r = open(data_r_file, "w")
    dr1 = open(datafolder + '/data_r{}{}_{}.txt'.format(num1, num2, num1), "w")
    dr2 = open(datafolder + '/data_r{}{}_{}.txt'.format(num1, num2, num2), "w")

    n0 = []
    n1 = []
    n2 = []
    n3 = []
    n4 = []
    n5 = []
    with open(edgefile, "r") as lines:
        for l in lines:
            n = l.strip("\n").split(" ")
            if int(n[0]) == num1:
                print(n[0], n[1], n[2], file=data)
                print(n[1], n[2], file=d1)
                n0.append(n[0])
                n1.append(n[1])
                n2.append(n[2])
            if int(n[0]) == num2:
                print(n[0], n[1], n[2], file=data)
                print(n[1], n[2], file=d2)
                n3.append(n[0])
                n4.append(n[1])
                n5.append(n[2])
    data.close()
    d1.close()
    d2.close()

    # “Specifically, for each real multiplex we select one of its layers and we interchange the ID of each node of the layer with the ID of a randomly selected node from the same layer.” ([Kleineberg 等, 2017, p. 2](zotero://select/library/items/N9TBK66C)) ([pdf](zotero://open-pdf/library/items/R548SG93?page=2&annotation=44GHJ6HE))
    # 🔤具体来说，对于每个真实多路复用器，我们选择其中的一个层，然后将该层每个节点的 ID 与从同一层随机选择的一个节点的 ID 互换。🔤
    # 构造映射
    remap = np.arange(0, N+1).astype(str)
    # shuffle(remap[1:N+1])
    remap = random_swap_elements(remap)

    if reshuffle_layer == num1:  # 选择 num1 层
        for i in range(0, len(n0)):
                n1[i] = remap[int(n1[i])]
                n2[i] = remap[int(n2[i])]

    if reshuffle_layer == num2:  # 选择 num2 层
        for j in range(0, len(n3)):
            n4[j] = remap[int(n4[j])]
            n5[j] = remap[int(n5[j])]

    for i in range(0, len(n0)):
        print(n1[i], n2[i], file=dr1)
        print(str(num1), n1[i], n2[i], file=data_r)
    for j in range(0, len(n3)):
        print(n4[j], n5[j], file=dr2)
        print(str(num2), n4[j], n5[j], file=data_r)
    data_r.close()
    dr1.close()
    dr2.close()

    layer_name = open(datafolder + '/name.txt', "w")
    layer_name.write(str(num1) + '\n')
    layer_name.write(str(num2))
    layer_name.close()

#datapath = '../../../data/'
datapath = 'multiplex_edges/'

node_num = {'Padgett-Florentine-Families_multiplex': 16,
            'AirTrain': 69,  # [(1,2)]
            'Brain': 90,  # [(1,2)]
            # 'fao_trade_multiplex': 214,
            'Phys': 246,  # [(1,2), (1,3), (2,3)]
            'celegans_connectome_multiplex': 279,  # [(1,2), (1,3), (2,3)]
            # 'HumanMicrobiome_multiplex': 305,
            # 'xenopus_genetic_multiplex': 416,
            # 'pierreauger_multiplex': 514,
            'rattus_genetic_multiplex': 2640,  # [(1,2)]
            'sacchpomb_genetic_multiplex': 4092,  # [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4),(3,5)]
            'drosophila_genetic_multiplex': 8215,  # [(1,2)]
            'arxiv_netscience_multiplex': 14489,  # [(1,2),(1,4),(1,5),(2,4),(2,5),(2,6),(2,8),(3,4)]
            'Internet': 4202010733}

nums_dict = {'AirTrain': [(1,2)],
             'Brain': [(1,2)],
             'Phys': [(2,3)],  # [(1,2), (1,3), (2,3)],
             'celegans_connectome_multiplex': [(2,3)],
             'rattus_genetic_multiplex': [(1,2)],
             'sacchpomb_genetic_multiplex': [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4),(3,5)],
             'drosophila_genetic_multiplex': [(1,2)],
             'arxiv_netscience_multiplex': [(1,2),(1,4),(1,5),(2,4),(2,5),(2,6),(2,8),(3,4)]}

if __name__ == "__main__":
    dataname = 'arxiv_netscience_multiplex'
    N = node_num[dataname]
    nums = nums_dict[dataname]
    for num in nums:
        num1 = num[0]
        num2 = num[1]
        print()
        print(dataname, '-', num1, '-', num2)
        datafolder = datapath + dataname
        pre_data(dataname, num1=num1, num2=num2, N=N, reshuffle_layer=num1)
        G1 = nx.Graph()
        G1.add_nodes_from(range(1, N + 1))
        G2 = nx.Graph()
        G2.add_nodes_from(range(1, N + 1))
        # 读取文件
        f1 = open(datafolder + '/data{}{}_{}.txt'.format(num1, num2, num1), "rb")
        G1 = add_edges_from_file(G1, f1)
        f2 = open(datafolder + '/data{}{}_{}.txt'.format(num1, num2, num2), "rb")
        G2 = add_edges_from_file(G2, f2)
        M = MCC(G1, G2)  # 计算初始MCC
        print("M =", M)
        dN, MCCs, ps = critical_number(G1, G2, N, M)
        print("dN =", dN)
        # 创建一个正方形的figure，指定figsize参数
        fig = plt.figure(figsize=(4, 4))
        fig = draw_percolation_transition(fig, dataname, MCCs, ps, num1, num2, type='g-', label='Original')


        print()
        dNrss = []
        for iter in range(1):

            G1 = nx.Graph()
            G1.add_nodes_from(range(1, N + 1))
            G2 = nx.Graph()
            G2.add_nodes_from(range(1, N + 1))
            # 读取文件
            f1 = open(datafolder + '/data_r{}{}_{}.txt'.format(num1, num2, num1), "rb")
            G1 = add_edges_from_file(G1, f1)
            f2 = open(datafolder + '/data_r{}{}_{}.txt'.format(num1, num2, num2), "rb")
            G2 = add_edges_from_file(G2, f2)
            M = MCC(G1, G2)  # 计算初始MCC
            print("M =", M)
            dNrs, MCCs, ps = critical_number(G1, G2, N, M)
            # print("dNrs = {} (iter={})".format(dNrs, iter+1))
            dNrss.append(dNrs)
            fig = draw_percolation_transition(fig, dataname, MCCs, ps, num1, num2, type='r--', label='Reshuffled')

        print("dNrs =", sum(dNrss)/len(dNrss))

        # 添加图例
        plt.legend(loc='best')  # 'best'自动选择最佳位置
        # 保存图像为文件
        fig.savefig(datafolder + '/MCC_vs_p({}_{}).jpg'.format(num1, num2))

        # 显示图形
        plt.show()



