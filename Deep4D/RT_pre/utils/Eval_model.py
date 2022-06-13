import torch
import torch.nn as nn
from tqdm import tqdm
from .get_mask import get_mask
import numpy as np


def eval_model(model, loader, device, min, max, norm):
    model.eval()  
    n_val = len(loader) 
    tot_loss = 0 
    tot_MAE = 0  
    tot = torch.tensor([]).to(device=device, dtype=torch.float32)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            peptide, ccs = batch['peptide'], batch['RT']
            peptide = peptide.to(device=device, dtype=torch.float32)
            ccs = ccs.to(device=device, dtype=torch.float32)
            length = batch['length'].to(device=device, dtype=torch.float32)
            if norm:
                    ccs = ccs/norm
            else:ccs = (ccs-min)/(max-min)
            mask = get_mask(peptide, length).to(device=device, dtype=torch.bool)
            peptide = peptide.transpose(0, 1)
            with torch.no_grad():  
                ccs_pred = model(src=peptide,src_key_padding_mask=mask).view(ccs.size(0))
            loss_f = nn.L1Loss()
            tot_loss += torch.abs(ccs-ccs_pred).mean() 
            tot_MAE += (torch.abs(ccs-ccs_pred)).mean()
            tot = torch.cat([tot,torch.abs(ccs-ccs_pred)],0)
            pbar.update()
    model.train() 
    tot = np.array(tot.to(device='cpu', dtype=torch.float32))

    return tot_loss/n_val , tot_MAE/n_val  