import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

z_to_matrix = {2:[1, 0, 0],
               3:[0, 1, 0],
               4:[0, 0, 1],
               5:[1, 1, 0],
               6:[1, 0, 1],
               7:[0, 1, 1],
               9:[1, 1, 1]}

def transfer_charge(charge,device):
    x = torch.zeros([charge.size(0),3]).to(device=device,dtype=torch.float32)
    for i in range(charge.size(0)):
        x[i,] = torch.tensor(z_to_matrix[int(charge[i])]).to(device=device, dtype=torch.float32)
    return  x

def get_mask(peptide,length):
    mask = torch.zeros(peptide.size(0),peptide.size(1))
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 1
    return  mask

def eval_model(model, loader, device):
    model.eval()  ##将model调整为eval模式
    n_val = len(loader)
    tot = torch.tensor([]).to(device=device, dtype=torch.float32)
    tot_pre = torch.tensor([]).to(device=device, dtype=torch.float32)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            peptide = batch['peptide_onehot'].to(device=device, dtype=torch.float32)
            length = batch['length'].to(device=device, dtype=torch.float32)
            mask = get_mask(peptide, length).to(device=device, dtype=torch.bool)
            charge = batch['charge'].to(device=device, dtype=torch.float32)
            charge = transfer_charge(charge,device)
            with torch.no_grad():
                charge_pred = model(src=peptide,src_key_padding_mask=mask)
            loss_f = nn.L1Loss()
            tot = torch.cat([tot, charge], 0)
            tot_pre = torch.cat([tot_pre, charge_pred], 0)
            pbar.update()
    model.train() ##将模型调回train模式
    tot = tot.ge(0.5)
    tot_pre = tot_pre.ge(0.5)
    tot = np.array(tot.to(device='cpu', dtype=torch.float32))
    tot_pre = np.array(tot_pre.to(device='cpu', dtype=torch.float32))
    num = 0
    for i in range(len(tot)):
        if (tot[i,] == tot_pre[i,]).all():
            num = num + 1
    acc = num/len(tot)
    return acc