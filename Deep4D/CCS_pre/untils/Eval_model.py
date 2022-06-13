import torch
import torch.nn as nn
from tqdm import tqdm
from .get_mask import get_mask
import numpy as np


def eval_model(model, loader, device, min, max, norm):
    model.eval()  ##将model调整为eval模式
    n_val = len(loader)  # val中的batch数量
    tot_loss = 0  ##relative error
    tot_ARE = 0  ##average relative error
    CCS = torch.tensor([]).to(device=device, dtype=torch.float32)
    CCS_pre = torch.tensor([]).to(device=device, dtype=torch.float32)

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            peptide, ccs = batch['peptide'], batch['CCS']
            peptide = peptide.to(device=device, dtype=torch.float32)
            ccs = ccs.to(device=device, dtype=torch.float32)
            length = batch['length'].to(device=device, dtype=torch.float32)
            if norm:
                ccs = ccs / norm
            else:
                ccs = (ccs - min) / (max - min)
            mask = get_mask(peptide, length).to(device=device, dtype=torch.bool)
            peptide = peptide.transpose(0, 1)
            charge = batch['charge'].to(device=device, dtype=torch.float32)
            with torch.no_grad():  ##不生成梯度，减少运算量
                ccs_pred = model(src=peptide, src_key_padding_mask=mask).view(ccs.size(0))
            loss_f = nn.L1Loss()
            CCS = torch.cat((CCS, ccs), 0)
            CCS_pre = torch.cat((CCS_pre, ccs_pred), 0)
            tot_loss += torch.abs(ccs - ccs_pred).mean()  
            pbar.update()
    model.train()  ##将模型调回train模式
    tot = torch.abs(CCS - CCS_pre) / CCS
    tot_MeanRE = tot.mean()
    tot = np.array(tot.to(device='cpu', dtype=torch.float32))
    tot_MedianRE = np.median(tot)
    return tot_loss / n_val, tot_MeanRE, tot_MedianRE  