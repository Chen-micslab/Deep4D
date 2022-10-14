import numpy as np
import pandas as pd
import os
import shutil
import argparse
from process_utils import filter_phos_ion
from aa_onehot import onehot

def get_args():  ##设置需要传入的参数
    parser = argparse.ArgumentParser(description='Train the transformer on peptide and ccs')
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--type', type=str, default='DeepDIA')
    parser.add_argument('--label', type=int, default=1)
    parser.add_argument('--maxcharge', type=int, default=4)
    parser.add_argument('--slice', type=int, default=1)
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
    if peptide[0] == 'a':  
        peptide_np[0, 20] = 1
    peptide_np[:pep_len, ] = peptide_np[:pep_len, ] + sinusoid_table[:pep_len, ]  
    return peptide_np

def DeepDIA_msms_encoding(pep_dataframe,peptide):   ####包含+1，+2价的b、y离子，中性丢失包含NH3和H20
    peptide = peptide.replace('a','')
    lenth = len(peptide)
    charge = np.array(pep_dataframe['FI.Charge'])
    by_type = np.array(pep_dataframe['FI.FrgType'])
    loss = np.array(pep_dataframe['FI.LossType'])
    by_id = np.array(pep_dataframe['FI.FrgNum'])
    intensity = np.array(pep_dataframe['FI.Intensity'])
    intensity = intensity/(np.max(intensity))
    data = np.zeros((49,12))
    data[(lenth-1):, ] = -1
    for i in range(len(charge)):
        if by_type[i] == 'b':
            if charge[i] == 1:
                if loss[i] == 'noloss':
                    data[(by_id[i] - 1), 0] = intensity[i]
                elif loss[i] == 'NH3':
                    data[(by_id[i] - 1), 1] = intensity[i]
                elif loss[i] == 'H20':
                    data[(by_id[i] - 1), 2] = intensity[i]
            elif charge[i] == 2:
                if loss[i] == 'noloss':
                    data[(by_id[i] - 1), 3] = intensity[i]
                elif loss[i] == 'NH3':
                    data[(by_id[i] - 1), 4] = intensity[i]
                elif loss[i] == 'H20':
                    data[(by_id[i] - 1), 5] = intensity[i]
        elif by_type[i] == 'y':
            if charge[i] == 1:
                if loss[i] == 'noloss':
                    data[(by_id[i] - 1), 6] = intensity[i]
                elif loss[i] == 'NH3':
                    data[(by_id[i] - 1), 7] = intensity[i]
                elif loss[i] == 'H20':
                    data[(by_id[i] - 1), 8] = intensity[i]
            elif charge[i] == 2:
                if loss[i] == 'noloss':
                    data[(by_id[i] - 1), 9] = intensity[i]
                elif loss[i] == 'NH3':
                    data[(by_id[i] - 1), 10] = intensity[i]
                elif loss[i] == 'H20':
                    data[(by_id[i] - 1), 11] = intensity[i]
    return data

def DeepPhospho_msms_encoding(pep_dataframe, peptide):   ####包含+1，+2价的b、y离子，中性丢失包含H3PO4
    peptide = peptide.replace('a','')
    lenth = len(peptide)
    charge = np.array(pep_dataframe['FI.Charge'])
    by_type = np.array(pep_dataframe['FI.FrgType'])
    loss = np.array(pep_dataframe['FI.LossType'])
    by_id = np.array(pep_dataframe['FI.FrgNum'])
    intensity = np.array(pep_dataframe['FI.Intensity'])
    intensity = intensity/(np.max(intensity))
    # data = np.full((49,8),-1,dtype=float)
    data= np.zeros((49,8))
    data[(lenth-1):,] = -1
    for i in range(len(charge)):
        if by_type[i] == 'b':
            if charge[i] == 1:
                if loss[i] == 'Noloss':
                    data[(by_id[i] - 1), 0] = intensity[i]
                elif loss[i] == '1,H3PO4':
                    data[(by_id[i] - 1), 1] = intensity[i]
            elif charge[i] == 2:
                if loss[i] == 'Noloss':
                    data[(by_id[i] - 1), 2] = intensity[i]
                elif loss[i] == '1,H3PO4':
                    data[(by_id[i] - 1), 3] = intensity[i]
        elif by_type[i] == 'y':
            if charge[i] == 1:
                if loss[i] == 'Noloss':
                    data[(by_id[i] - 1), 4] = intensity[i]
                elif loss[i] == '1,H3PO4':
                    data[(by_id[i] - 1), 5] = intensity[i]
            elif charge[i] == 2:
                if loss[i] == 'Noloss':
                    data[(by_id[i] - 1), 6] = intensity[i]
                elif loss[i] == '1,H3PO4':
                    data[(by_id[i] - 1), 7] = intensity[i]
    data = filter_phos_ion(peptide, data)
    return data


def encoding_with_label(input, output, maxcharge, datatype):
    data = pd.read_csv(input)
    os.mkdir(f'{output}_msms')
    os.mkdir(f'{output}_onehot')
    charge_list = list(range(2,(maxcharge+1)))  ###eg: maxcharge = 4, charge_list = [2, 3, 4]
    num = 0
    sinusoid_table = np.array([get_position_angle_vec(pos_i, 23) for pos_i in range(50)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  
    for z in charge_list:
        data1 = data[data['Charge'] == z]
        if len(np.array(data1['Peptide'])) > 0:
            data1 = data1.sort_values('Peptide', ignore_index=True)  
            peptide_list = np.array(data1['Peptide'])
            name = list(data1)
            peptide = peptide_list[0]
            index_list = []
            peptide_num = len(set(data1['Peptide']))
            for i in range(len(peptide_list)):
                if peptide_list[i] == peptide:
                    index_list.append(i)
                else:
                    num = num + 1
                    if num%200 == 0:
                        print(num)
                    data2 = data1.iloc[index_list]
                    if peptide[0] == 'a':
                        pep_len = len(peptide) - 1
                    else: pep_len = len(peptide)
                    if len(peptide) < 50:
                        if datatype == 'DeepDIA':
                            msms_arrary = DeepDIA_msms_encoding(data2, peptide).ravel()
                        elif datatype == 'DeepPhospho':
                            msms_arrary = DeepPhospho_msms_encoding(data2, peptide).ravel()
                        onehot_arrary = peptide_onehot_encoding(peptide, sinusoid_table)
                        num_id = str(num).zfill(7)
                        b = msms_arrary
                        b[b==-1]=0
                        if np.sum(b) != 0:  ####去除二级谱图为空的
                            np.save(f'{output}_msms/{num_id}_{peptide}_${pep_len}$#{z}#.npy', msms_arrary)
                            np.save(f'{output}_onehot/{num_id}_{peptide}_${pep_len}$#{z}#.npy', onehot_arrary)
                    index_list = []
                    index_list.append(i)
                    peptide = peptide_list[i]
            num = num + 1
            if peptide[0] == 'a':
                pep_len = len(peptide) - 1
            else:
                pep_len = len(peptide)
            if len(peptide) < 50:
                if datatype == 'DeepDIA':
                    msms_arrary = DeepDIA_msms_encoding(data2, peptide).ravel()
                elif datatype == 'DeepPhospho':
                    msms_arrary = DeepPhospho_msms_encoding(data2, peptide).ravel()
                onehot_arrary = peptide_onehot_encoding(peptide, sinusoid_table)
                num_id = str(num).zfill(7)
                b = msms_arrary
                b[b == -1] = 0
                if np.sum(b) != 0:  ####去除二级谱图为空的
                    np.save(f'{output}_msms/{num_id}_{peptide}_${pep_len}$#{z}#.npy', msms_arrary)
                    np.save(f'{output}_onehot/{num_id}_{peptide}_${pep_len}$#{z}#.npy', onehot_arrary)
                        
def encoding_without_label(input, output, maxcharge, slice = 1):  ###这里是只对肽做onehot编码
    data_total = pd.read_csv(input)
    name = list(data_total)
    total_lenth = len(list(data_total['Peptide']))
    data_total = np.array(data_total)
    offset = 0
    sinusoid_table = np.array([get_position_angle_vec(pos_i, 23) for pos_i in range(50)])  # 先计算position encoding sin和cos里面的值
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i  按照偶数来算sin
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1  按照奇数算cos
    for slice_num in range(slice):
        print(f'{slice_num}/{slice} slices in encoding..........')
        if slice_num != (slice - 1):
            lenth1 = total_lenth // slice
        else:
            lenth1 = total_lenth // slice + total_lenth % slice
        data = data_total[offset:(offset + lenth1),:]
        offset = offset + lenth1
        data = pd.DataFrame(data, columns=name)
        os.mkdir(f'{output}_slice{slice_num}_onehot')
        charge_list = list(range(2,(maxcharge+1)))  ###eg: maxcharge = 4, charge_list = [2, 3, 4]
        num = 0
        for z in charge_list:
            data1 = data[data['Charge'] == z]
            peptide_list = data1['Peptide']
            for peptide in peptide_list:
                num = num + 1
                if peptide[0] == 'a':
                    pep_len = len(peptide) - 1
                else: pep_len = len(peptide)
                onehot_arrary = peptide_onehot_encoding(peptide, sinusoid_table)
                num_id = str(num).zfill(7)
                if pep_len<50:
                    np.save(f'{output}_slice{slice_num}_onehot/{num_id}_{peptide}_${pep_len}$#{z}#.npy', onehot_arrary)
                    
if __name__ == '__main__':
    args = get_args()
    if args.label == 1:
        encoding_with_label(f'./data/{args.filename}.csv', f'./data/{args.filename}', args.maxcharge, args.type)
    else:
        encoding_without_label(f'./data/{args.filename}.csv', f'./data/{args.filename}', args.maxcharge, args.slice)
