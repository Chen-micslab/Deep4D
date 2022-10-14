import argparse
import sys
import os
from datetime import datetime
sys.path.append('./program')
sys.path.append('./dataset')
import numpy as  np
import pandas as pd
import torch
from tqdm import tqdm

from  predict_msms1 import predict_nolabel_msms
from  predict_ccs import predict_nolabel_ccs
from  predict_rt import predict_nolabel_rt
from  dataset.Encoding_msms import encoding_without_label as encoding_msms
from  program.Generate_spectral_library import merge_information, generate_4d_library, change_to_peaks

def get_args():  ##设置需要传入的参数
    parser = argparse.ArgumentParser(description='Predict 4D library')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--feature_len', type=int, default=23)
    parser.add_argument('--d_model', type=int, default=500)
    parser.add_argument('--nheads', type=int, default=5)
    parser.add_argument('--num_encoder_layers', type=int, default=5)
    parser.add_argument('--dim_feedforward', type=int, default=1200)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--maxcharge', type=int, default=3)  ###所预测的最高价态
    parser.add_argument('--msms_norm', type=float, default=10)  
    parser.add_argument('--ccs_norm', type=float, default=100)  
    parser.add_argument('--rt_norm', type=float, default=10)  
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--load_msms_param_dir', type=str, default='./checkpoint/raw_msms.pth')
    parser.add_argument('--load_ccs_param_dir', type=str, default='./checkpoint/raw_ccs.pth')
    parser.add_argument('--load_rt_param_dir', type=str, default='./checkpoint/raw_rt.pth')
    parser.add_argument('--seed', type=int, default=1)  
    parser.add_argument('--label', type=int, default=0)  ### 1:label, 0:nolabel
    parser.add_argument('--filename', type=str, default='raw_trypsin_charge23')  ###输入文件，为csv格式，包含Peptide和Charge两列信息
    parser.add_argument('--type', type=str, default='DeepDIA')  ###二级谱图的构成为何种格式
    parser.add_argument('--slice', type=int, default='3')  ###防止内存不足无法处理所有的peptide，把所有peptide分为slice个部分，分别预测，最后再汇总
    parser.add_argument('--Merged_predict', type=bool, default=True)  ###是否同时预测msms，rt，ccs
    parser.add_argument('--library', type=str, default='Peaks')  ###生成的library为何种格式
    return parser.parse_args()

def create_taskname(filename):
    a = str(datetime.now())
    a = a.replace(':', '-')
    a = a.replace('.', '-')
    a = a.replace(' ', '-')
    a = a[:19]
    task_name = f'{filename}_task_{a}'
    return task_name

if __name__ == '__main__':
    args = get_args()
    task_name = create_taskname(args.filename)
    os.mkdir(f'./dataset/data/{task_name}')
    os.mkdir(f'./dataset/data/{task_name}/output')
    data_dir = f'./dataset/data/{args.filename}.csv'
    task_dir = f'./dataset/data/{task_name}'
    onehot_dir = f'./dataset/data/{task_name}/{args.filename}'
    encoding_msms(data_dir, onehot_dir, args.maxcharge, args.slice)
    predict_nolabel_msms(args, task_dir)
    predict_nolabel_ccs(args, task_dir)
    predict_nolabel_rt(args, task_dir)
    print('merging prediction information.........')
    for slice_num in range(args.slice):
        msms_dir = f'./dataset/data/{task_name}/output/{args.filename}_slice{slice_num}_pre_msms.csv'
        ccs_dir = f'./dataset/data/{task_name}/output/{args.filename}_slice{slice_num}_pre_ccs.csv'
        rt_dir = f'./dataset/data/{task_name}/output/{args.filename}_slice{slice_num}_pre_rt.csv'
        print(f'{slice_num} in slice{args.slice}: merging 4d information...........')
        data = merge_information(msms_dir, ccs_dir, rt_dir)
        print(f'{slice_num} in slice{args.slice}: generating 4d library...........')
        data1 = generate_4d_library(data, args.type)
        data1 = np.array(data1)
        if slice_num == 0:
            data_lib = data1
        else:
            data_lib = np.row_stack((data_lib, data1))
    name = ['Peptide','Charge','m_z','RT','IM','FI.Charge','FI.FrgType','FI.FrgNum','FI.LossType','FI.Intensity','FI.m_z']
    data_lib = pd.DataFrame(data_lib, columns=name)
    data_lib.to_csv(f'./dataset/data/{task_name}/output/{args.filename}_normal_library.csv',index=False)
    if args.library == 'Peaks':
        print('transfering to Peaks library.........')
        a = change_to_peaks()
        data_lib = a.forward(data_lib)
        data_lib.to_csv(f'./dataset/data/{task_name}/output/{args.filename}_Peaks_library.tsv',index=False, sep='\t')
