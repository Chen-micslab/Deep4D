import argparse
import  logging
import os
import sys

import numpy as  np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from dataset.Dataset_ccs import Mydata_label, Mydata_nolabel
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.Eval_ccs_model import eval_model
from model.selfatt_cnn_z import Transformer

#######这个程序适用于预测的肽特别多的时候，先将csv文件里的所有肽转换成包含npy文件的文件夹，然后在用这个程序

def get_args():  ##设置需要传入的参数
    parser = argparse.ArgumentParser(description='Predict ccs')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--feature_len', type=int, default=23)
    parser.add_argument('--d_model', type=int, default=500)
    parser.add_argument('--nheads', type=int, default=5)
    parser.add_argument('--num_encoder_layers', type=int, default=5)
    parser.add_argument('--dim_feedforward', type=int, default=1200)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--ccs_norm', type=float, default=100) 
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--output_dir', type=str, default='./output')  ##
    parser.add_argument('--load_ccs_param_dir', type=str, default=None)  
    parser.add_argument('--seed', type=int, default=1)  
    parser.add_argument('--label', type=int, default=1)  ### 1:label, 0:nolabel
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--slice', type=int, default='1')
    return parser.parse_args()

def get_mask(peptide,length): 
    mask = torch.zeros(peptide.size(0),peptide.size(1))  
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 0 
    return  mask

def predict_label_ccs(args):
    data_dir = f'./dataset/data/{args.filename}'
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ##判断使用GPU还是CPU
    model = Transformer(feature_len=args.feature_len,
                        d_model=args.d_model,
                        nhead=args.nheads,
                        num_encoder_layers=args.num_encoder_layers,
                        dim_feedforward=args.dim_feedforward,
                        dropout=args.dropout,
                        activation=args.activation)
    if args.load_ccs_param_dir:
        model.load_state_dict(torch.load(args.load_ccs_param_dir, map_location=device))
    model.to(device=device)
    model.eval()  ##将model调整为eval模式
    test_data = Mydata_label(data_dir)
    total_lenth = len(test_data)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=3, pin_memory=True)
    n_val = len(test_loader)  # val中的batch数量
    norm = args.ccs_norm
    index = 0
    total_peptide = np.zeros(total_lenth, dtype=object)
    total_peptide_charge = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
    CCS = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
    CCS_pre = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
    with tqdm(total=n_val, desc='Prediction round', unit='batch', leave=False) as pbar:
        for batch in test_loader:
            peptide = batch['peptide_onehot'].to(device=device, dtype=torch.float32)
            ccs = batch['CCS'].to(device=device, dtype=torch.float32)
            ccs  = ccs/norm
            length = batch['length'].to(device=device, dtype=torch.float32)
            charge = batch['charge'].to(device=device, dtype=torch.float32)
            charge = charge.unsqueeze(-1)
            peptide_seq = np.array(batch['peptide'])
            mask = get_mask(peptide, length).to(device=device, dtype=torch.bool)
            with torch.no_grad():  ##不生成梯度，减少运算量
                ccs_pre = model(src=peptide, src_key_padding_mask=mask, charge = charge).view(ccs.size(0))
                id = ccs.size(0)
                total_peptide[index:(index+id)] = peptide_seq
                CCS[index:(index + id)] = ccs
                CCS_pre[index:(index+id)] = ccs_pre
                charge = charge.squeeze(-1)
                total_peptide_charge[index:(index+id)] = charge
                index = index + id
            pbar.update()
    tot = torch.abs(CCS - CCS_pre) / CCS
    tot = np.array(tot.to(device='cpu', dtype=torch.float32))
    print('MAE:', np.mean(tot))
    print('MRE:', np.median(tot))
    CCS = CCS.unsqueeze(1) * norm
    CCS_pre = CCS_pre.unsqueeze(1) * norm
    total_peptide_charge = total_peptide_charge.unsqueeze(1)
    total_peptide = np.expand_dims(np.array(total_peptide), axis=1)
    data_total = torch.cat((total_peptide_charge, CCS, CCS_pre), 1)
    data_total = data_total.to(device='cpu', dtype=torch.float32).numpy()
    data = np.column_stack((total_peptide,data_total))
    data = pd.DataFrame(data, columns=['Peptide', 'Charge', 'CCS', 'CCS_pre'])
    data.to_csv(f'./dataset/data/output/{args.filename}_ccs_pre.csv', index=False)  # 保存所有的预测值和真实值到一个CSV文件

def predict_nolabel_ccs(args):
    slice = args.slice
    for slice_num in range(slice):
        print(f'{slice_num}/{slice} slices')
        data_dir = f'../LC_pre/dataset/data/{args.filename}{slice_num}'
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ##判断使用GPU还是CPU
        model = Transformer(feature_len=args.feature_len,
                            d_model=args.d_model,
                            nhead=args.nheads,
                            num_encoder_layers=args.num_encoder_layers,
                            dim_feedforward=args.dim_feedforward,
                            dropout=args.dropout,
                            activation=args.activation)
        if args.load_ccs_param_dir:
            model.load_state_dict(torch.load(args.load_ccs_param_dir, map_location=device))
        model.to(device=device)
        model.eval()  ##将model调整为eval模式
        test_data = Mydata_nolabel(data_dir)
        total_lenth = len(test_data)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=3, pin_memory=True)
        n_val = len(test_loader)  # val中的batch数量
        norm = args.ccs_norm
        index = 0
        total_peptide = np.zeros(total_lenth, dtype=object)
        total_peptide_charge = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
        CCS_pre = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
        with tqdm(total=n_val, desc='Prediction round', unit='batch', leave=False) as pbar:
            for batch in test_loader:
                peptide = batch['peptide_onehot'].to(device=device, dtype=torch.float32)
                length = batch['length'].to(device=device, dtype=torch.float32)
                charge = batch['charge'].to(device=device, dtype=torch.float32)
                peptide_seq = np.array(batch['peptide'])
                mask = get_mask(peptide, length).to(device=device, dtype=torch.bool)
                with torch.no_grad():  ##不生成梯度，减少运算量
                    ccs_pre = model(src=peptide, src_key_padding_mask=mask, charge = charge).view(charge.size(0))
                    id = charge.size(0)
                    total_peptide[index:(index + id)] = peptide_seq
                    CCS_pre[index:(index + id)] = ccs_pre
                    total_peptide_charge[index:(index + id)] = charge
                    index = index + id
                pbar.update()
        CCS_pre = CCS_pre.unsqueeze(1) * norm
        total_peptide_charge = total_peptide_charge.unsqueeze(1)
        total_peptide = np.expand_dims(np.array(total_peptide), axis=1)
        data_total = torch.cat((total_peptide_charge, CCS_pre), 1)
        data_total = data_total.to(device='cpu', dtype=torch.float32).numpy()
        data = np.column_stack((total_peptide, data_total))
        data = pd.DataFrame(data, columns=['Peptide', 'Charge', 'CCS_pre'])
        data.to_csv(f'../LC_pre/dataset/data/output/{args.filename}{slice_num}_ccs_pre.csv', index=False)  # 保存所有的预测值和真实值到一个CSV文件
if __name__ == '__main__':
    args = get_args()  ##生成参数列表
    if args.label == 1:
        predict_label_ccs(args)
    else:
        predict_nolabel_ccs(args)
