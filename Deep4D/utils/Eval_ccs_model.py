import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def get_mask(peptide,length):
    mask = torch.zeros(peptide.size(0),peptide.size(1))  
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 1 
    return  mask

def eval_model(model, loader, n_val, device, norm):
    model.eval()  ##将model调整为eval模式
    total_lenth = n_val
    CCS = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
    CCS_pre = torch.zeros(total_lenth).to(device=device, dtype=torch.float32)
    index = 0
    with tqdm(total=len(loader), desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            peptide = batch['peptide_onehot'].to(device=device, dtype=torch.float32)
            ccs = batch['CCS'].to(device=device, dtype=torch.float32)
            ccs = ccs / norm
            length = batch['length'].to(device=device, dtype=torch.float32)
            charge = batch['charge'].to(device=device, dtype=torch.float32)
            mask = get_mask(peptide, length).to(device=device, dtype=torch.bool) 
            with torch.no_grad():  ##不生成梯度，减少运算量
                ccs_pre = model(src=peptide, src_key_padding_mask=mask, charge = charge).view(ccs.size(0))
                id = ccs.size(0)
                CCS[index:(index + id)] = ccs
                CCS_pre[index:(index+id)] = ccs_pre
                index = index + id
            pbar.update()
    tot = torch.abs(CCS - CCS_pre) / CCS
    tot_MeanRE = tot.mean()
    tot = np.array(tot.to(device='cpu', dtype=torch.float32))
    tot_MedianRE = np.median(tot)
    return tot_MeanRE, tot_MedianRE