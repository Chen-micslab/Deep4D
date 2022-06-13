import numpy as np
import pandas as pd
import os
import shutil


def extract(str,symbol):  
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

def get_onehot(input,output):
    data = pd.read_csv(input, header=0)  
    os.mkdir(output) 
    data = np.array(data)  
    pepetide_num = len(data[:, 0]) 
    for h in range(pepetide_num):
        if h % 200 == 0:
            print(h)
        peptide_np = np.zeros((50, 23))  
        peptide = data[h, 0]
        peptide_ccs = int(data[h, 2] * 10000)  
        peptide_charge = int(data[h, 1])  
        if peptide[0] == 'a': 
            x = 1
            pep_len = len(peptide) - 1
        else:
            x = 0
            pep_len = len(peptide)
        i = 0
        for j in range(x, len(peptide)):
            if peptide[j] == 'A':  # 丙氨酸
                peptide_np[i, :] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'R':  # 精氨酸
                peptide_np[i, :] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'D':  # 天冬氨酸
                peptide_np[i, :] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'C':  # 半胱氨酸
                peptide_np[i, :] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'Q':  # 谷氨酰胺
                peptide_np[i, :] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'E':  # 谷氨酸
                peptide_np[i, :] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'H':  # 组氨酸
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'I':  # 异亮氨酸
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'G':  # 甘氨酸
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'N':  # 天冬酰胺
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'L':  # 亮氨酸
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'K':  # 赖氨酸
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'M':  # 甲硫氨酸
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'F':  # 苯丙氨酸
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'P':  # 脯氨酸
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'S':  # 丝氨酸
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'T':  # 苏氨酸
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            elif peptide[j] == 'W':  # 色氨酸
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            elif peptide[j] == 'Y':  # 酪氨酸
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            elif peptide[j] == 'V':  # 缬氨酸
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            elif peptide[j] == 's':  # b=phos--S, c=phos--T, d=phos--Y, e=Oxidation-M
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
            elif peptide[j] == 't':  # b=phos--S, c=phos--T, d=phos--Y, e=Oxidation-M
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
            elif peptide[j] == 'y':  # b=phos--S, c=phos--T, d=phos--Y, e=Oxidation-M
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
            elif peptide[j] == 'e':  # b=phos--S, c=phos--T, d=phos--Y, e=Oxidation-M
                peptide_np[i, :] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            i = i + 1
        if peptide[0] == 'a':  
            peptide_np[0, 20] = 1
        sinusoid_table = np.array([get_position_angle_vec(pos_i,23) for pos_i in range(pep_len)]) 
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  
        peptide_np[:pep_len,] = peptide_np[:pep_len,] + sinusoid_table 
        lenth = len(str(pepetide_num))
        num = str(h).zfill(lenth)
        np.save(f'{output}/{num}_{peptide_ccs}_${pep_len}$#{peptide_charge}#.npy',
                peptide_np)  

