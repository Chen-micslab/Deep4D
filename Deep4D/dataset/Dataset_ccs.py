import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from os.path import splitext
from os import listdir
import os
import torch.nn as nn
import  math
import torch.nn.functional as F

class Mydata_label(Dataset):
    def __init__(self,datadir):
        self.onehot_dir = f'{datadir}_onehot' ##这里data_dir是文件的储存路径
        self.allpeptide = listdir(self.onehot_dir)  ##储存每一个peptide的文件名
        self.lenth = len(str(len(self.allpeptide)))  ###获得肽总数的位数
        self.allpeptide.sort(key=lambda x: int(x[:self.lenth]))  # 将所有文件名，按照序号从小到大排序

    def extract(self,str, symbol):  ###定义一个函数来提取相同的开始符和结束符中间的字符串,symbol可以设置为任意字符
        index = []
        for i in range(len(str)):
            if str[i] == symbol:
                index.append(i)
        start_index = index[0]
        end_index = index[1]
        str1 = str[(start_index + 1):end_index]
        return str1

    def __getitem__(self, index):
        peptide_index = self.allpeptide[index]  ##按照index提取任意peptide的文件名
        peptide_onehot_dir = os.path.join(self.onehot_dir,peptide_index) ##拼接文件名和路径
        peptide_onehot = np.load(peptide_onehot_dir)  ##按照peptide的文件路径读取peptide的one-hot矩阵
        CCS = int(self.extract(peptide_index,'@'))/10000  
        peptide_len = int(self.extract(peptide_index,'$'))
        peptide_charge = int(self.extract(peptide_index, '#'))  
        peptide_sep = self.extract(peptide_index, '_')
        sample = {'peptide_onehot':peptide_onehot, 'length':peptide_len,'charge':peptide_charge, 'peptide':peptide_sep,'CCS':CCS}
        return sample

    def __len__(self):
        return len(self.allpeptide)

class Mydata_nolabel(Dataset):
    def __init__(self, datadir):
        self.onehot_dir = f'{datadir}_onehot'  ##这里data_dir是文件的储存路径
        self.allpeptide = listdir(self.onehot_dir)  ##储存每一个peptide的文件名
        self.lenth = len(str(len(self.allpeptide)))  ###获得肽总数的位数
        self.allpeptide.sort(key=lambda x: int(x[:self.lenth]))  # 将所有文件名，按照序号从小到大排序

    def extract(self, str, symbol):  ###定义一个函数来提取相同的开始符和结束符中间的字符串,symbol可以设置为任意字符
        index = []
        for i in range(len(str)):
            if str[i] == symbol:
                index.append(i)
        start_index = index[0]
        end_index = index[1]
        str1 = str[(start_index + 1):end_index]
        return str1

    def __getitem__(self, index):
        peptide_index = self.allpeptide[index]  ##按照index提取任意peptide的文件名
        peptide_onehot_dir = os.path.join(self.onehot_dir, peptide_index)  ##拼接文件名和路径
        peptide_onehot = np.load(peptide_onehot_dir)  ##按照peptide的文件路径读取peptide的one-hot矩阵
        peptide_len = int(self.extract(peptide_index, '$'))  
        peptide_charge = int(self.extract(peptide_index, '#'))  
        peptide_sep = self.extract(peptide_index, '_')
        sample = {'peptide_onehot': peptide_onehot, 'length': peptide_len, 'charge': peptide_charge, 'peptide': peptide_sep}
        return sample

    def __len__(self):
        return len(self.allpeptide)





