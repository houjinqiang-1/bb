from FINDER_protect import FINDER
import numpy as np
import time
import os
import pandas as pd
import torch.backends.cudnn as cudnn
import torch
import random
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

g_type = "GMM"


def GetSolution(STEPRATIO, MODEL_FILE):
    ######################################################################################################################
    ##................................................Get Solution (model).....................................................
    dqn = FINDER()
    data_test_path = './data/real/'
    data_test_name = ['celegans_connectome_multiplex','CS-Aarhus_multiplex','fao_trade_multiplex','fb-tw','homo_genetic_multiplex','sacchpomb_genetic_multiplex','Sanremo2016_final_multiplex']
    date_test_n = [279,61,214,1043,18222,4092,56562]
    data_test_layer = [(2,3),(1,5),(3,24),(1,2),(1,2),(4,6),(2,3)]
    data_test_name = ['Sanremo2016_final_multiplex']
    date_test_n = [56562]
    data_test_layer = [(1,2)]
    data_test_name = ['us_air_transportation_american_delta_multiplex',
                      'drosophila_melanogaster_multiplex','netsci_co-authorship_multiplex']
    date_test_n = [84,557,1400]
    data_test_layer = [(1,2),(5,6),(1,2)]
    model_file = './models/{}'.format(MODEL_FILE)
    ## save_dir
    save_dir = '../results/FINDER_ND_weightQ/protect_FINDER/g0.5'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    ## begin computing...
    print ('The best model is :%s'%(model_file))
    dqn.LoadModel(model_file)
    for j in range(len(data_test_name)):
        df = pd.DataFrame(np.arange(2*len(data_test_name)).reshape((2,len(data_test_name))),index=['time','score'], columns=data_test_name)
        #################################### modify to choose which stepRatio to get the solution
        stepRatio = STEPRATIO
        print ('\nTesting dataset %s'%data_test_name[j])
        #data_test = data_test_path + data_test_name[j] + '.txt'
        data_test = data_test_path + data_test_name[j] + '.edges'
        #solution, time = dqn.EvaluateRealData(model_file, data_test, save_dir, stepRatio)
        solution, time, score= dqn.EvaluateRealData(model_file, data_test, save_dir, stepRatio,date_test_n[j],data_test_layer[j])
        df.iloc[0,j] = time
        df.iloc[1,j] = score
        print('Data:%s, time:%.2f, score:%.6f'%(data_test_name[j], time, score))
        save_dir_local = save_dir + '/StepRatio_%.4f' % stepRatio
        if not os.path.exists(save_dir_local):
            os.mkdir(save_dir_local)
        df.to_csv(save_dir_local + '/sol_time&score_%s.csv'% data_test_name[j], encoding='utf-8', index=False)


def main():
    model_file_ckpt = 'g0.5_TORCH-Model_{}_30_50/nrange_30_50_iter_100000.ckpt'.format(g_type)
    GetSolution(0, model_file_ckpt)


if __name__=="__main__":
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    main()
