import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import numpy.linalg as L
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import pandas as pd

def get_cosine(msms, msms_pre):
    dot = np.dot(msms,msms_pre)
    return dot/(L.norm(msms)*L.norm(msms_pre))

def get_SA(msms, msms_pre):
    L2normed_act = msms / L.norm(msms)
    L2normed_pred = msms_pre / L.norm(msms_pre)
    inner_product = np.dot(L2normed_act, L2normed_pred)
    return 1 - 2*np.arccos(inner_product)/np.pi

def get_pearson(act, pred):
    return pearsonr(act, pred)[0]


def get_spearman(act, pred):
    return spearmanr(act, pred)[0]

def get_mask(peptide,length):
    mask = torch.zeros(peptide.size(0),peptide.size(1))  
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 1 
    return  mask

def eval_model(model, loader,total_lenth, device, norm, type):
    model.eval()  ##将model调整为eval模式
    n_val = len(loader)  # val中的batch数量
    num = 0
    index = 0
    if type == 'DeepPhospho':
        feature_len = 392
    elif type == 'DeepDIA':
        feature_len = 588
    total_peptide_msms = torch.zeros((total_lenth,feature_len)).to(device=device, dtype=torch.float32)
    total_peptide_msms_pre = torch.zeros((total_lenth,feature_len)).to(device=device, dtype=torch.float32)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            peptide = batch['peptide_onehot'].to(device=device, dtype=torch.float32)
            length = batch['length'].to(device=device, dtype=torch.float32)
            peptide_msms = batch['peptide_msms'].to(device=device, dtype=torch.float32)
            charge = batch['charge'].to(device=device, dtype=torch.float32)
            mask = get_mask(peptide, length).to(device=device, dtype=torch.bool)
            peptide_msms = norm * peptide_msms
            with torch.no_grad():  ##不生成梯度，减少运算量
                peptide_msms_pre = model(src=peptide, src_key_padding_mask=mask,charge=charge)
                id = peptide_msms_pre.size(0)
                total_peptide_msms[index:(index+id),:] = peptide_msms
                total_peptide_msms_pre[index:(index+id),:] = peptide_msms_pre
                index = index + id
            pbar.update()
    model.train()  ##将模型调回train模式
    total_peptide_msms = np.array(total_peptide_msms.to(device='cpu', dtype=torch.float32))
    total_peptide_msms_pre = np.array(total_peptide_msms_pre.to(device='cpu', dtype=torch.float32))
    cosine = []
    SA = []
    pearson = []
    spearman = []
    x = -1 * norm
    for i in range(len(total_peptide_msms)):
        a = total_peptide_msms[i, :]
        b = total_peptide_msms_pre[i, :]
        # select = (a != 0) * (a != x)
        select =  (a != x)
        if len(a[(a != 0) * (a != x)]) > 2:
            print(b[select])
            cosine.append(get_cosine(a[select], b[select]))
            SA.append(get_SA(a[select], b[select]))
            pearson.append(get_pearson(a[select], b[select]))
            spearman.append(get_spearman(a[select], b[select]))
    mean_cosine = np.mean(cosine)
    median_cosine = np.median(cosine)
    mean_SA = np.mean(SA)
    median_SA = np.median(SA)
    mean_pearson = np.mean(pearson)
    median_pearson = np.median(pearson)
    mean_spearman = np.mean(spearman)
    median_spearman = np.median(spearman)
    return mean_cosine, median_cosine, mean_SA, median_SA, mean_pearson, median_pearson, mean_spearman, median_spearman