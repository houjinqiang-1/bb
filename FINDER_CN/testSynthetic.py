7#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from FINDER_torch import FINDER
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch
import random
import numpy as np
def main():
    dqn = FINDER()
    data_test_path = '../data/synthetic/uniform_cost/'
    #data_test_name = ['30-50', '50-100', '100-200', '200-300', '300-400', '400-500']
    # data_test_name =['32','64','128','256','512','1024']
    data_test_name = ['1024']
    model_file = './models/g0.5_TORCH-Model_GMM_30_50/nrange_30_50_iter_100000.ckpt'
    types = ['data_g', 'data_gamma', 'data_k']
    for data_type in types:
        file_path = f'./results/MultDismantler/synthetic_cost/{data_type}_1'
        if not os.path.exists(file_path):
             os.makedirs(file_path, exist_ok=True)
        for i in tqdm(range(len(data_test_name))):
            with open('%s/result_%s_unit_cost.txt'%(file_path,data_test_name[i]), 'w') as fout:
                data_test = data_test_path + data_test_name[i]
                score_mean, score_std, time_mean, time_std, cost_mean = dqn.Evaluate(data_test, data_test_name[i], data_type, model_file)
                # fout.write('%.4f±%.2f,' % (score_mean , score_std ))
                fout.write('%.4f' % (cost_mean))
                fout.flush()
                print('data_test_%s has been tested!' % data_test_name[i])


if __name__=="__main__":
    # cudnn.benchmark = True
    # cudnn.deterministic = False
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    main()

