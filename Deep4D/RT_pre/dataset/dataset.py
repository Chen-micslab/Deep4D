import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from os.path import splitext
from os import listdir
import os
import torch.nn as nn
import  math
import torch.nn.functional as F

class Mydata(Dataset):

    def __init__(self,data_dir):
        self.data_dir = data_dir 
        self.allpeptide = listdir(self.data_dir)  
        self.lenth = len(str(len(self.allpeptide)))  
        self.allpeptide.sort(key=lambda x: int(x[:self.lenth]))  


    def extract(self,str, symbol):  
        index = []
        for i in range(len(str)):
            if str[i] == symbol:
                index.append(i)
        start_index = index[0]
        end_index = index[1]
        str1 = str[(start_index + 1):end_index]
        return int(str1)

    def __getitem__(self, index):
        peptide_index = self.allpeptide[index]  
        peptide_dir = os.path.join(self.data_dir,peptide_index) 
        peptide = np.load(peptide_dir)  
        RT = self.extract(peptide_index,'_')/10000  
        peptide_len = self.extract(peptide_index,'$') 
        sample = {'peptide':peptide,'RT':RT,'length':peptide_len}
        return sample

    def __len__(self):
        return len(self.allpeptide)





