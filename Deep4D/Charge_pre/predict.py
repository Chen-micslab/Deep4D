import argparse
import  logging
import os
import sys
import shutil
import numpy as  np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from pathlib import Path
import time
from model import selfatt_cnn_charge
from dataset.dataset import Mydata
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.Eval_model_charge import eval_model
from model.selfatt_cnn_charge import Transformer
from dataset.one_hot_positional_encoding import get_onehot
from utils.get_mask import get_mask

def get_args():  
    parser = argparse.ArgumentParser(description='Predict charge state')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--feature_len', type=int, default=23)
    parser.add_argument('--d_model', type=int, default=500)
    parser.add_argument('--nheads', type=int, default=5)
    parser.add_argument('--num_encoder_layers', type=int, default=5)
    parser.add_argument('--dim_feedforward', type=int, default=1200)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--norm', type=float, default=1)  
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--output_dir', type=str, default='./output')  
    parser.add_argument('--load_param_dir', type=str,
                        default=None)  
    parser.add_argument('--seed', type=int, default=1)  
    parser.add_argument('--filename', type=str, default=None)
    return parser.parse_args()


def extract(str,symbol):  
    index = []
    for i in range(len(str)):
        if str[i]==symbol:
            index.append(i)
    start_index = index[0]
    end_index = index[1]
    str1 = str[(start_index+1):end_index]
    return str1

def get_position_angle_vec(position,dim): 
    return [position / np.power(10000, 2 * (hid_j // 2) / dim) for hid_j in range(dim)]


if __name__ == '__main__':

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
   args = get_args()
   data_csvdir =f'./dataset/data/input/{args.filename}.csv'  
   data_csv = pd.read_csv(data_csvdir)
   data_npydir = './temporary'  
   if Path(data_npydir).exists():
       shutil.rmtree(data_npydir)
   get_onehot(data_csvdir,data_npydir)  
   torch.cuda.manual_seed(args.seed)
   torch.manual_seed(args.seed)
   np.random.seed(args.seed)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
   model = Transformer(feature_len=args.feature_len,
                       d_model=args.d_model,
                       nhead=args.nheads,
                       num_encoder_layers=args.num_encoder_layers,
                       dim_feedforward=args.dim_feedforward,
                       dropout=args.dropout,
                       activation=args.activation)
   if args.load_param_dir:
       model.load_state_dict(torch.load(args.load_param_dir, map_location=device))
   model.to(device=device)
   model.eval()  
   test_data = Mydata(data_npydir)
   test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=3, pin_memory=True)
   n_val = len(test_loader)  
   norm = args.norm
   charge_pre = torch.tensor([]).to(device=device, dtype=torch.float32)
   time_start = time.time()
   with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
       for batch in test_loader:
           peptide = batch['peptide']
           peptide = peptide.to(device=device, dtype=torch.float32)      
           length = batch['length'].to(device=device, dtype=torch.float32)
           mask = get_mask(peptide, length).to(device=device, dtype=torch.bool)
           peptide = peptide.transpose(0, 1)
           with torch.no_grad():  
               z_pred = model(src=peptide,src_key_padding_mask=mask)
           charge_pre = torch.cat((charge_pre, z_pred), 0)
           loss_f = nn.L1Loss()
           pbar.update()
   model.train()  
   shutil.rmtree(data_npydir)
   charge_pre = charge_pre.ge(0.5)
   charge_pre = charge_pre.to(device='cpu', dtype=torch.float32).numpy()
   charge_pre = pd.DataFrame(charge_pre,columns=['charge2','charge3','charge4'])
   charge_pre.to_csv(f'./dataset/data/output/{args.filename}_pre_z.csv',index=False)  
   time_end = time.time()
   print('totally cost', time_end - time_start)

