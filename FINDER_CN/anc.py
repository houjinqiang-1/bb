import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch
import random
import testReal
import os

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
             'Phys': [(1,2), (1,3), (2,3)],  # [(1,2), (1,3), (2,3)],
             'celegans_connectome_multiplex': [(2,3)],
             'rattus_genetic_multiplex': [(1,2)],
             'sacchpomb_genetic_multiplex': [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4),(3,5)],
             'drosophila_genetic_multiplex': [(1,2)],
             'arxiv_netscience_multiplex': [(1,4)], #[(1,2)]#[(1,2),(1,4),(1,5),(2,4),(2,5),(2,6),(2,8),(3,4)]
             'Padgett-Florentine-Families_multiplex':[(1,2)]
             }

#画出HDA和OUR的ANC图
def judge():
    data_name = 'AirTrain'
    num1 = 1
    num2 = 2
    path = './ANC/'+ data_name + f'/{data_name}-{num1}-{num2}'
    MCC1 = np.loadtxt('../results/FINDER_CN/real/StepRatio_0.0010/MaxCCList_Strategy_ACM3025.txt', usecols=(0,))
    # 将MCC1保存到txt文件中
    np.savetxt(path+'/our_mcc.txt', MCC1, fmt='%f')
    solution = np.loadtxt('../results/FINDER_CN/real/StepRatio_0.0010/ACM3025.txt',usecols=(0,))
    np.savetxt(path + '/our_solution.txt', solution, fmt='%d')
    MCC2 = np.load(path + f'/{data_name}-{num1}-{num2}' + '.npy')[:-1]
    # 计算填充数量
    fill_count = len(MCC2) - len(MCC1)
    # 扩展MCC1的长度并填充
    if fill_count > 0:
        fill_value = MCC1[-1]  # 使用MCC1的最后一个值填充
        MCC1 = np.append(MCC1, np.full(fill_count, fill_value))
    x1 = np.ones(len(MCC1))
    for i in range(0,len(MCC1)):
        x1[i] = i/len(MCC1)
    x2 = np.ones(len(MCC2))
    for i in range(0,len(MCC2)):
        x2[i] = i/len(MCC2)
    plt.figure()
    plt.plot(x1,MCC1,label='MRGNN_FINDER')
    plt.plot(x2,MCC2,label='ANC')
    plt.ylabel('Value')
    plt.title(f'data:{data_name}-{num1}-{num2},iter:best radio:0.001')
    plt.legend()
    # 设置 x 轴范围为 0 到 0.2
    plt.xlim(0, 1)
    plt.savefig(f'{path}/img_best.png')
    plt.show()

testReal.main()
judge()
