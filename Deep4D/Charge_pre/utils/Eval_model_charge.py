import torch
import torch.nn as nn
from tqdm import tqdm
from .get_mask import get_mask
import numpy as np

def transfer_charge(charge,device):
    x = torch.zeros([charge.size(0),3]).to(device=device,dtype=torch.float32)
    for i in range(charge.size(0)):
        if charge[i] == 2:
            x[i,] = torch.tensor([1, 0, 0]).to(device=device,dtype=torch.float32)
        elif charge[i] == 3:
            x[i,] = torch.tensor([0, 1, 0]).to(device=device, dtype=torch.float32)
        elif charge[i] == 4:
            x[i,] = torch.tensor([0, 0, 1]).to(device=device, dtype=torch.float32)
        elif charge[i] == 5:
            x[i,] = torch.tensor([1, 1, 0]).to(device=device, dtype=torch.float32)
        elif charge[i] == 6:
            x[i,] = torch.tensor([1, 0, 1]).to(device=device, dtype=torch.float32)
        elif charge[i] == 7:
            x[i,] = torch.tensor([0, 1, 1]).to(device=device, dtype=torch.float32)
        elif charge[i] == 9:
            x[i,] = torch.tensor([1, 1, 1]).to(device=device, dtype=torch.float32)
    return  x

def eval_model(model, loader, device, min, max):
    model.eval()  
    n_val = len(loader)  
    tot = torch.tensor([]).to(device=device, dtype=torch.float32)
    tot_pre = torch.tensor([]).to(device=device, dtype=torch.float32)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            peptide = batch['peptide']
            peptide = peptide.to(device=device, dtype=torch.float32)
            length = batch['length'].to(device=device, dtype=torch.float32)
            mask = get_mask(peptide, length).to(device=device, dtype=torch.bool)
            peptide = peptide.transpose(0, 1)
            charge = batch['charge'].to(device=device, dtype=torch.float32)
            charge = transfer_charge(charge,device)
            with torch.no_grad():  
                charge_pred = model(src=peptide,src_key_padding_mask=mask)
            loss_f = nn.L1Loss()
            tot = torch.cat([tot, charge], 0)
            tot_pre = torch.cat([tot_pre, charge_pred], 0)
            pbar.update()
    model.train() 
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