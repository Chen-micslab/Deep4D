import argparse
import  logging
import os
import sys
sys.path.append('./program')
import time
import numpy as  np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from dataset.Dataset_msms import Mydata_label, Mydata_nolabel, Subset
from torch.utils.data import DataLoader
from utils.Eval_msms_model import eval_model, get_cosine, get_SA, get_pearson, get_spearman
from dataset.process_utils import filter_phos_ion
from program.mass_cal import regular_peptide_m_z

def get_args():  ##设置需要传入的参数
    parser = argparse.ArgumentParser(description='Predict msms')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--feature_len', type=int, default=23)
    parser.add_argument('--d_model', type=int, default=500)
    parser.add_argument('--nheads', type=int, default=5)
    parser.add_argument('--num_encoder_layers', type=int, default=5)
    parser.add_argument('--dim_feedforward', type=int, default=1200)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--msms_norm', type=float, default=10)  ##CCS除以的常数
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--output_dir', type=str, default='./output')  
    parser.add_argument('--load_msms_param_dir', type=str, default=None)  
    parser.add_argument('--seed', type=int, default=1)  
    parser.add_argument('--filename', type=str, default='normalpeptide_5%FR_total_charge4_test')
    parser.add_argument('--label', type=int, default=1)  ### 1:label, 0:nolabel
    parser.add_argument('--type', type=str, default='DeepDIA')
    parser.add_argument('--slice', type=int, default='1')
    return parser.parse_args()

def get_mask(peptide,length): 
    mask = torch.zeros(peptide.size(0),peptide.size(1))  
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 1
    return  mask

def predict_label_msms(args):
    data_dir = f'./dataset/data/{args.filename}'
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ##判断使用GPU还是CPU
    if args.type == 'DeepDIA':
        from model.selfatt_cnn_deepdia import Transformer as deep_model
    elif args.type == 'DeepPhospho':
        from model.selfatt_cnn_deepphos import Transformer as deep_model
    model = deep_model(feature_len=args.feature_len,
                        d_model=args.d_model,
                        nhead=args.nheads,
                        num_encoder_layers=args.num_encoder_layers,
                        dim_feedforward=args.dim_feedforward,
                        dropout=args.dropout,
                        activation=args.activation)
    if args.load_msms_param_dir:
        model.load_state_dict(torch.load(args.load_msms_param_dir, map_location=device))
    model.to(device=device)
    model.eval()  ##将model调整为eval模式
    test_data = Mydata_label(data_dir)
    total_lenth = len(test_data)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=3, pin_memory=True)
    n_val = len(test_loader)  # val中的batch数量
    norm = args.msms_norm
    num = 0
    x = -1 * norm
    index = 0
    total_peptide = np.zeros(total_lenth, dtype=object)
    total_peptide_charge = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
    total_peptide_lenth = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
    if args.type == 'DeepDIA':
        total_peptide_msms = torch.zeros((total_lenth, 588)).to(device=device, dtype=torch.float32)
        total_peptide_msms_pre = torch.zeros((total_lenth, 588)).to(device=device, dtype=torch.float32)
    elif args.type == 'DeepPhospho':
        total_peptide_msms = torch.zeros((total_lenth, 392)).to(device=device, dtype=torch.float32)
        total_peptide_msms_pre = torch.zeros((total_lenth, 392)).to(device=device, dtype=torch.float32)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in test_loader:
            peptide = batch['peptide_onehot'].to(device=device, dtype=torch.float32)
            length = batch['length'].to(device=device, dtype=torch.float32)
            peptide_msms = batch['peptide_msms'].to(device=device, dtype=torch.float32)
            charge = batch['charge'].to(device=device, dtype=torch.float32)
            mask = get_mask(peptide, length).to(device=device, dtype=torch.bool)  
            peptide_seq = np.array(batch['peptide'])
            peptide_msms = norm * peptide_msms
            with torch.no_grad():  ##不生成梯度，减少运算量
                peptide_msms_pre = model(src=peptide, src_key_padding_mask=mask, charge=charge)  
                id = peptide_msms_pre.size(0)
                total_peptide[index:(index+id)] = peptide_seq
                total_peptide_msms[index:(index + id), :] = peptide_msms
                total_peptide_msms_pre[index:(index+id),:] = peptide_msms_pre
                total_peptide_charge[index:(index+id)] = charge
                total_peptide_lenth[index:(index+id)] = length
                index = index + id
            pbar.update()
    model.train()  ##将模型调回train模式
    total_peptide_msms_pre[torch.where(total_peptide_msms == x)] = x
    total_peptide_msms = np.array(total_peptide_msms.to(device='cpu', dtype=torch.float32))
    total_peptide_msms_pre = np.array(total_peptide_msms_pre.to(device='cpu', dtype=torch.float32))
    total_peptide_charge = np.array(total_peptide_charge.to(device='cpu', dtype=torch.int))
    total_peptide_lenth = np.array(total_peptide_lenth.to(device='cpu', dtype=torch.int))
    cosine = []
    SA = []
    pearson = []
    spearman = []
    pep = []
    charge_list = []
    for i in range(len(total_peptide_msms)):
        a = total_peptide_msms[i, :]
        b = total_peptide_msms_pre[i, :]
        # select = (a != 0) * (a != x)
        select = (a != 0) * (a != x)
        if len(a[(a != 0) * (a != x)]) > 2:
            pep.append(total_peptide[i])
            charge_list.append(total_peptide_charge[i])
            cosine.append(get_cosine(a[select], b[select]))
            SA.append(get_SA(a[select], b[select]))
            pearson.append(get_pearson(a[select], b[select]))
            spearman.append(get_spearman(a[select], b[select]))
    data = np.column_stack((cosine, SA, pearson, spearman))
    pep = np.expand_dims(np.array(pep), axis=1)
    charge_list = np.expand_dims(np.array(charge_list), axis=1)
    data = np.column_stack((pep, charge_list, data))
    data = pd.DataFrame(data, columns=['Peptide', 'Charge', 'cosine', 'SA', 'pearson', 'spearman'])
    data.to_csv(f'./dataset/data/output/{args.filename}_pre_perform.csv', index=False)
    print('Validation Mean cosine: {}'.format(np.mean(cosine)))
    print('Validation Median cosine: {}%'.format(np.median(cosine)))
    print('Validation Mean SA: {}'.format(np.mean(SA)))
    print('Validation Median SA: {}%'.format(np.median(SA)))
    print('Validation Mean pearson: {}'.format(np.mean(pearson)))
    print('Validation Median pearson: {}%'.format(np.median(pearson)))
    print('Validation Mean spearman: {}'.format(np.mean(spearman)))
    print('Validation Median spearman: {}%'.format(np.median(spearman)))
    num = 0
    pep = []
    m_z_list = []
    charge_list = []
    pep_len = []
    total_peptide_msms_pre_1 = total_peptide_msms_pre
    for i in range(len(total_peptide_msms)):
        a = total_peptide_msms[i, :]
        b = total_peptide_msms_pre[i, :]
        if len(b[(b != 0) * (b != x)]) > 2:
            b = b / np.max(b)
            m_z = regular_peptide_m_z(total_peptide[i], total_peptide_charge[i])
            m_z_list.append(m_z)
            pep.append(total_peptide[i])
            charge_list.append(total_peptide_charge[i])
            pep_len.append(total_peptide_lenth[i])
            total_peptide_msms_pre_1[i,:] = b
            num = num + 1
    total_peptide_msms_pre_1 = total_peptide_msms_pre_1[:num, :]
    pep = np.expand_dims(np.array(pep), axis=1)
    m_z_list = np.expand_dims(np.array(m_z_list), axis=1)
    charge_list = np.expand_dims(np.array(charge_list), axis=1)
    pep_len = np.expand_dims(np.array(pep_len), axis=1)
    data = np.column_stack((pep, charge_list, m_z_list, pep_len, total_peptide_msms_pre_1))
    data = pd.DataFrame(data)
    data.to_csv(f'./dataset/data/output/{args.filename}_pre.csv', index=False)

def predict_nolabel_msms(args):
    data_dir = f'./dataset/data/{args.filename}'
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ##判断使用GPU还是CPU
    if args.type == 'DeepDIA':
        from model.selfatt_cnn_deepdia import Transformer as deep_model
    elif args.type == 'DeepPhospho':
        from model.selfatt_cnn_deepphos import Transformer as deep_model
    model = deep_model(feature_len=args.feature_len,
                        d_model=args.d_model,
                        nhead=args.nheads,
                        num_encoder_layers=args.num_encoder_layers,
                        dim_feedforward=args.dim_feedforward,
                        dropout=args.dropout,
                        activation=args.activation)
    if args.load_msms_param_dir:
        model.load_state_dict(torch.load(args.load_msms_param_dir, map_location=device))
    model.to(device=device)
    model.eval()  ##将model调整为eval模式
    test_data = Mydata_nolabel(data_dir)
    total_lenth = len(test_data)
    indices = list(range(total_lenth))
    slice = args.slice
    offset = 0
    for slice_num in range(slice):
        print(f'{slice_num}/{slice} slices')
        if slice_num != (slice - 1):
            lenth1 = total_lenth // slice
        else:
            lenth1 = total_lenth // slice + total_lenth % slice
        test_data_1 = Subset(test_data, indices[offset:(offset + lenth1)])
        offset = offset + lenth1

        test_loader = DataLoader(test_data_1, batch_size=args.batch_size, shuffle=False, num_workers=3, pin_memory=True)
        norm = args.norm
        x = -1 * norm
        index = 0
        total_peptide = np.zeros(total_lenth, dtype=object)
        total_peptide_charge = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
        total_peptide_lenth = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
        if args.type == 'DeepDIA':
            total_peptide_msms_pre = torch.zeros((total_lenth,588)).to(device=device, dtype=torch.float32)
        elif args.type == 'DeepPhospho':
            total_peptide_msms_pre = torch.zeros((total_lenth,392)).to(device=device, dtype=torch.float32)
        with tqdm(total=(int(len(test_data_1)/args.batch_size)), desc='Prediction round', unit='batch', leave=False) as pbar:
            for batch in test_loader:
                peptide = batch['peptide_onehot'].to(device=device, dtype=torch.float32)
                length = batch['length'].to(device=device, dtype=torch.float32)
                charge = batch['charge'].to(device=device, dtype=torch.float32)
                mask = get_mask(peptide, length).to(device=device, dtype=torch.bool)  ##求相应的mask矩阵
                peptide_seq = np.array(batch['peptide'])
                with torch.no_grad():  ##不生成梯度，减少运算量
                    peptide_msms_pre = model(src=peptide, src_key_padding_mask=mask, charge=charge)  ##将数据送入model，得到预测的ccs
                    id = peptide_msms_pre.size(0)
                    total_peptide[index:(index+id)] = peptide_seq
                    total_peptide_msms_pre[index:(index+id),:] = peptide_msms_pre
                    total_peptide_charge[index:(index+id)] = charge
                    total_peptide_lenth[index:(index+id)] = length
                    index = index + id
                pbar.update()
        total_peptide_msms_pre = np.array(total_peptide_msms_pre.to(device='cpu', dtype=torch.float32))
        total_peptide_charge = np.array(total_peptide_charge.to(device='cpu', dtype=torch.int))
        total_peptide_lenth = np.array(total_peptide_lenth.to(device='cpu', dtype=torch.int))
        num = 0
        pep = []
        charge_list = []
        m_z_list = []
        pep_len = []
        total_peptide_msms_pre_1 = total_peptide_msms_pre
        for i in range(len(total_peptide_msms_pre)):
            b = total_peptide_msms_pre[i, :]
            b[8*total_peptide_lenth[i]:] = x
            if args.type == 'DeepPhospho':
                b = b.reshape(())
                b = filter_phos_ion(total_peptide[i], b)
                b = b.ravel()
            if len(b[(b > 0)]) > 2:  ###保留起码含有三个碎片离子的谱图
                b = b / np.max(b)
                m_z = regular_peptide_m_z(total_peptide[i], total_peptide_charge[i])
                m_z_list.append(m_z)
                pep.append(total_peptide[i])
                charge_list.append(total_peptide_charge[i])
                pep_len.append(total_peptide_lenth[i])
                total_peptide_msms_pre_1[i,:] = b
                num = num + 1
        total_peptide_msms_pre_1 = total_peptide_msms_pre_1[:num, :]
        pep = np.expand_dims(np.array(pep), axis=1)
        m_z_list = np.expand_dims(np.array(m_z_list), axis=1)
        charge_list = np.expand_dims(np.array(charge_list), axis=1)
        pep_len = np.expand_dims(np.array(pep_len), axis=1)
        data = np.column_stack((pep, charge_list, m_z_list, pep_len, total_peptide_msms_pre_1))
        data = pd.DataFrame(data)
        data.to_csv(f'./dataset/data/output/{args.filename}_pre_slice{slice_num}.csv', index=False)


if __name__ == '__main__':
    args = get_args()  ##生成参数列表
    if args.label == 1:
        predict_label_msms(args)
    else:
        predict_nolabel_msms(args)

