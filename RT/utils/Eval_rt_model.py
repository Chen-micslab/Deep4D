import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def get_mask(peptide,length):
    mask = torch.zeros(peptide.size(0),peptide.size(1))  
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 1 
    return  mask

def eval_model(model, loader,n_val, device, norm):
    model.eval()  ##将model调整为eval模式
    RT = torch.zeros(n_val).to(device=device, dtype=torch.float32)
    RT_pre = torch.zeros(n_val).to(device=device, dtype=torch.float32)
    index = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            peptide = batch['peptide_onehot'].to(device=device, dtype=torch.float32)
            rt = batch['rt'].to(device=device, dtype=torch.float32)
            rt  = rt/norm
            length = batch['length'].to(device=device, dtype=torch.float32)
            charge = batch['charge'].to(device=device, dtype=torch.float32)
            mask = get_mask(peptide,length).to(device=device,dtype=torch.bool)  
            with torch.no_grad():  ##不生成梯度，减少运算量
                rt_pre = model(src=peptide,src_key_padding_mask=mask).view(rt.size(0))
                id = rt.size(0)
                RT[index:(index + id)] = rt
                RT_pre[index:(index+id)] = rt_pre
                index = index + id
            pbar.update()
    tot_ARE = (torch.abs(RT - RT_pre)).mean()
    return  tot_ARE*norm