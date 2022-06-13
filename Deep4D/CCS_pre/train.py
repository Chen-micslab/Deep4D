import argparse
import  logging
import os
import sys

import numpy as  np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from model import selfatt_cnn
from dataset.dataset import Mydata
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.Eval_model import eval_model
from model.selfatt_cnn import Transformer
from utils.get_mask import get_mask


def get_args():  
    parser = argparse.ArgumentParser(description='Train the Deep4D on peptide and ccs')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--feature_len', type=int, default=23)
    parser.add_argument('--d_model', type=int, default=500)
    parser.add_argument('--nheads', type=int, default=5)
    parser.add_argument('--num_encoder_layers', type=int, default=5)
    parser.add_argument('--dim_feedforward', type=int, default=1200)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max', type=float, default=629.5419871)
    parser.add_argument('--min', type=float, default=272.0595023)
    parser.add_argument('--norm', type=float, default=100)  
    parser.add_argument('--validation', type=float, default=0.1) 
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--output_dir', type=str, default='./output')  
    parser.add_argument('--load_param_dir', type=str, default=None) 
    parser.add_argument('--seed', type=int, default=1)  
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--sch', type=int, default=0)
    return parser.parse_args()

def get_lr(epoch):  
    if epoch<3:
        lr = 1
    elif 3<=epoch<8:
        lr = 0.2
    elif 8<=epoch<20:
        lr = 0.04
    elif 20<=epoch<30:
        lr = 0.2
    elif 30<=epoch<50:
        lr = 0.04
    else:lr = 0.004
    return lr

def get_position_angle_vec(position, dim):  
    return [position / np.power(10000, 2 * (hid_j // 2) / dim) for hid_j in range(dim)]


def get_position_coding(max_len, d_model):
    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i, d_model) for pos_i in range(max_len)])  
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  
    return sinusoid_table

def train(model,
          device,
          epochs=10,
          batch_size=1,
          lr=0.001,
          val_percent=0.1,
          save_mp=True,  
          min=0,
          max=700,
          sch = 0
          ):

    mydata = Mydata(data_dir)  
    n_val = int(len(mydata) * val_percent)  
    n_train = len(mydata) - n_val   
    train_data, val_data = random_split(mydata ,[n_train, n_val])  
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)  
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True) 
    pos_1 = torch.from_numpy(get_position_coding(50, 500)).to(device='cuda', dtype=torch.float32)
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0
    print(max,min)
    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_mp}
            Device:          {device.type}
        ''')
    optimizer =optim.Adam( model.parameters(), lr=lr, betas=(0.9,0.999),eps=1e-08, weight_decay=0, amsgrad=False) 
    if sch==0:
         scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=10,T_mult=2,eta_min=0.00000001)
    else:
         scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=get_lr)
    for epoch in range(epochs):
        model.train()  
        local_step = 0 
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=get_lr(epoch))
        with tqdm(total=n_train, desc=f'Epoch {epoch+1}/{epochs}',unit='peptide') as pbar: 
            for batch in train_loader:
                local_step += 1
                peptide = batch['peptide']
                ccs = batch['CCS']
                length = batch['length']
                peptide = peptide.to(device=device,dtype=torch.float32) 
                ccs = ccs.to(device=device,dtype=torch.float32) 
                charge = batch['charge'].to(device=device,dtype=torch.float32)
                norm = args.norm
                if norm:
                    ccs = ccs/norm
                else:ccs = (ccs-min)/(max-min)
                charge = charge
                mask = get_mask(peptide,length).to(device=device,dtype=torch.bool)  
                peptide = peptide.transpose(0,1)  
                pos = pos_1.unsqueeze(1).repeat(1,peptide.size(1),1)
                ccs_pred = model(src=peptide,src_key_padding_mask=mask).view(ccs.size(0)) 
                print(ccs_pred.size())
                loss_f = nn.MSELoss()
                loss = loss_f(ccs,ccs_pred)  
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step=global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})  

                optimizer.zero_grad()  
                loss.backward()  
                optimizer.step()

                pbar.update(peptide.shape[0])
                global_step += 1
                if global_step % (n_train // (2 * batch_size)) == 0:
                    val_loss,val_MeanRE,val_MedianRE = eval_model(model,val_loader,device,min,max,norm)
                    logging.info('Validation MSEloss: {}'.format(val_loss))
                    logging.info('Validation MeanREloss: {}%'.format(val_MeanRE * 100))
                    logging.info('Validation MedianREloss: {}%'.format(val_MedianRE * 100))
        scheduler.step()
        if save_mp:
            logging.info('Created checkpoint directory')
            torch.save(model.state_dict(), checkpoint_dir + f'model_param_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
    writer.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') 
    args = get_args()  
    data_dir = f'./dataset/data/{args.filename}' 
    checkpoint_dir = f'./checkpoint/{args.filename}/'  
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(device)
    logging.info(f'Using device {device}')

    model = Transformer(feature_len=args.feature_len,
                        d_model=args.d_model,
                        nhead=args.nheads,
                        num_encoder_layers=args.num_encoder_layers,
                        dim_feedforward=args.dim_feedforward,
                        dropout=args.dropout,
                        activation=args.activation)  

    logging.info(f'Model:\n'
                 f'\tfeature_len is {args.feature_len}\n'
                 f'\td_model is {args.d_model}\n'
                 f'\targs.nheads is {args.nheads}\n'
                 f'\targs.num_encoder_layers is {args.num_encoder_layers}\n'
                 f'\targs.dim_feedforward is {args.dim_feedforward}\n'
                 f'\targs.dropout is {args.dropout}\n'
                 f'\targs.activation is {args.activation}\n')
    if args.load_param_dir:
        model.load_state_dict(torch.load(args.load_param_dir, map_location=device))
        logging.info(f'Model parameters loaded from {args.load_param_dir}')
    model.to(device=device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    train(model=model,
          device=device,
          epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.lr,
          val_percent=args.validation,
          save_mp=True,
          min = args.min,
          max = args.max,
          sch = args.sch
              )
