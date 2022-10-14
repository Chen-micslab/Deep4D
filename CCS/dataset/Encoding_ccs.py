import numpy as np
import pandas as pd
import os
import argparse
import shutil
from aa_onehot import onehot

def get_args():  ##设置需要传入的参数
    parser = argparse.ArgumentParser(description='Train the transformer on peptide and ccs')
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--label', type=int, default=1)
    return parser.parse_args()

def extract(str,symbol):  ###定义一个函数来提取相同的开始符和结束符中间的字符串
    index = []
    for i in range(len(str)):
        if str[i]==symbol:
            index.append(i)
    start_index = index[0]
    end_index = index[1]
    str1 = str[(start_index+1):end_index]
    return str1

def get_position_angle_vec(position,dim): 
    return [position / np.power(10000, 2 * (hid_j // 2) / dim) for hid_j in range(dim)]

def peptide_onehot_encoding(peptide, sinusoid_table):
    peptide_np = np.zeros((50, 23))  ###创建一个50*23的0矩阵来存放peptide
    if peptide[0] == 'a':  
        x = 1
        pep_len = len(peptide) - 1
    else:
        x = 0
        pep_len = len(peptide)
    i = 0
    for j in range(x, len(peptide)):
        peptide_np[i, :] = onehot.AA_onehot[peptide[j]]
        i = i + 1
    if peptide[0] == 'a':  ##再次判断如果第一位是a（乙酰化N端修饰），那么对one-hot矩阵的第一个氨基酸加入乙酰化的信息。
        peptide_np[0, 20] = 1
    peptide_np[:pep_len, ] = peptide_np[:pep_len, ] + sinusoid_table[:pep_len, ] 
    return peptide_np

def encoding_without_label(input, output): 
    data = pd.read_csv(input)
    os.mkdir(f'{output}_onehot')
    num = 0
    peptide_list = data['Peptide']
    charge_list = data['Charge']
    sinusoid_table = np.array([get_position_angle_vec(pos_i, 23) for pos_i in range(50)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  
    for i in range(len(peptide_list)):
        num = num + 1
        if num % 200 == 0:
            print(num)
        if peptide_list[i][0] == 'a':
            pep_len = len(peptide_list[i]) - 1
        else: pep_len = len(peptide_list[i])
        onehot_arrary = peptide_onehot_encoding(peptide_list[i], sinusoid_table)
        z = charge_list[i]
        num_id = str(num).zfill(7)
        if pep_len<50:
            np.save(f'{output}_onehot/{num_id}_{peptide_list[i]}_${pep_len}$#{z}#.npy', onehot_arrary)

def encoding_with_label(input, output):  
    data = pd.read_csv(input)
    os.mkdir(f'{output}_onehot')
    num = 0
    peptide_list = data['Peptide']
    charge_list = data['Charge']
    ccs_list = data['CCS']
    sinusoid_table = np.array([get_position_angle_vec(pos_i, 23) for pos_i in range(50)])  
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  
    for i in range(len(peptide_list)):
        num = num + 1
        if num % 200 == 0:
            print(num)
        if peptide_list[i][0] == 'a':
            pep_len = len(peptide_list[i]) - 1
        else: pep_len = len(peptide_list[i])
        onehot_arrary = peptide_onehot_encoding(peptide_list[i], sinusoid_table)
        z = charge_list[i]
        ccs = int(ccs_list[i] * 10000)
        num_id = str(num).zfill(7)
        if pep_len<50:
            np.save(f'{output}_onehot/{num_id}_{peptide_list[i]}_${pep_len}$#{z}#.npy', onehot_arrary)

if __name__ == '__main__':
    args = get_args()
    if args.label == 1:
        encoding_with_label(f'./data/{args.filename}.csv',f'./data/{args.filename}')
    else:
        encoding_without_label(f'./data/{args.filename}.csv',f'./data/{args.filename}')
