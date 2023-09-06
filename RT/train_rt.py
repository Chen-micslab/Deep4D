import argparse
import  logging
import os
import sys

import numpy as  np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from dataset.Dataset_rt import Mydata_label
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.Eval_rt_model import eval_model
from model.selfatt_cnn_rt import Transformer


def get_args():  ##设置需要传入的参数
    parser = argparse.ArgumentParser(description='Train the RT model')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--feature_len', type=int, default=23)
    parser.add_argument('--d_model', type=int, default=500)
    parser.add_argument('--nheads', type=int, default=5)
    parser.add_argument('--num_encoder_layers', type=int, default=5)
    parser.add_argument('--dim_feedforward', type=int, default=1200)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--rt_norm', type=float, default=10)  
    parser.add_argument('--validation', type=int, default=0)  ##是否存在validation数据集,1有，0无
    parser.add_argument('--vali_rate', type=float, default=0.1)  ##validation占总训练数据的比例
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--output_dir', type=str, default='./output')  ##
    parser.add_argument('--load_msms_param_dir', type=str, default=None)
    parser.add_argument('--load_ccs_param_dir', type=str, default=None)
    parser.add_argument('--load_rt_param_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)  
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--sch', type=int, default=0)
    parser.add_argument('--expect_performance', type=float, default=None)  ##validation占总训练数据的比例
    return parser.parse_args()

def get_position_angle_vec(position, dim):  ##position encoding,sin和cos函数里面的值
    return [position / np.power(10000, 2 * (hid_j // 2) / dim) for hid_j in range(dim)]


def get_position_coding(max_len, d_model):
    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i, d_model) for pos_i in range(max_len)])  
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  
    return sinusoid_table

def get_mask(peptide,length): 
    mask = torch.zeros(peptide.size(0),peptide.size(1)) 
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 0 
    return  mask

###设置模型训练时的细节
def train(model,device,epochs=10, batch_size=1,lr=0.001,val_percent=0.1,save_mp=True, sch = 0,
          traindir=None, validir=None, checkpoint_dir=None, expect_performance = None):
    if validir == None:
        mydata = Mydata_label(traindir)  ##导入dataset
        n_val = int(len(mydata) * val_percent)  ##计算validation data的数量
        n_train = len(mydata) - n_val  ##计算train data的数量
        train_data, val_data = random_split(mydata, [n_train, n_val])  ##随机分配validation data和train data
    else:
        train_data = Mydata_label(traindir)  ##导入dataset
        val_data = Mydata_label(validir)  ##导入dataset
        n_val = len(val_data) ##计算validation data的数量
        n_train = len(train_data)  ##计算train data的数量
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0,pin_memory=True)  ##生成train data
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,)  ##生成validation data
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0
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
         scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.2)
    best_index = 10
    for epoch in range(epochs):
        model.train()  
        local_step = 0
        epoch_loss = 0  
        with tqdm(total=n_train, desc=f'Epoch {epoch+1}/{epochs}',unit='peptide') as pbar: ##使用tqdm来显示训练的进度条
            for batch in train_loader:
                local_step += 1
                peptide = batch['peptide_onehot'].to(device=device, dtype=torch.float32)
                rt = batch['rt'].to(device=device, dtype=torch.float32)
                norm = args.rt_norm
                rt  = rt/norm
                length = batch['length'].to(device=device, dtype=torch.float32)
                charge = batch['charge'].to(device=device, dtype=torch.float32)
                mask = get_mask(peptide,length).to(device=device,dtype=torch.bool)  ##求相应的mask矩阵
                rt_pred = model(src=peptide,src_key_padding_mask=mask).view(rt.size(0)) ##将数据送入model，得到预测的ccs
                loss_f = nn.MSELoss()
                loss = loss_f(rt,rt_pred)  
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step=global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})  
                optimizer.zero_grad()  ##梯度清零
                loss.backward()  ##反向传播
                optimizer.step()
                pbar.update(peptide.shape[0])
                global_step += 1
                if global_step % (n_train // (2 * batch_size)) == 0:
                    val_MAE = eval_model(model,val_loader,n_val,device,norm)
                    writer.add_scalar('learning rate',optimizer.param_groups[0]['lr'], global_step)
                    logging.info('Validation MAE: {}'.format(val_MAE))
                    if val_MAE < best_index:
                        torch.save(model.state_dict(),
                                   checkpoint_dir + f'model_param_epoch{epoch + 1}global_step{global_step}val_MAE{val_MAE}.pth') 
                        logging.info(f'Checkpoint {epoch + 1}global_step{global_step} saved !')
                        best_index = val_MAE
        scheduler.step()
        if save_mp:
            logging.info('Created checkpoint directory')
            torch.save(model.state_dict(), checkpoint_dir + f'model_param_epoch{epoch + 1}.pth')###只保存模型的参数到checkpoint_dir
            logging.info(f'Checkpoint {epoch + 1} saved !')
    writer.close()

def do_train(args):
    if args.validation != 0:
        traindir = f'./dataset/data/{args.filename}_train'
        validir = f'./dataset/data/{args.filename}_validation'
    else:
        traindir = f'./dataset/data/{args.filename}'
        validir = None
    checkpoint_dir = f'./checkpoint/{args.filename}_rt/'  ##checkpoint文件夹的路径，用于保存模型参数
    if os.path.exists(checkpoint_dir):
        pass
    else: os.mkdir(checkpoint_dir)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ##判断使用GPU还是CPU
    print(device)
    logging.info(f'Using device {device}')

    model = Transformer(feature_len=args.feature_len,
                        d_model=args.d_model,
                        nhead=args.nheads,
                        num_encoder_layers=args.num_encoder_layers,
                        dim_feedforward=args.dim_feedforward,
                        dropout=args.dropout,
                        activation=args.activation)  ###生成模型

    logging.info(f'Model:\n'
                 f'\tfeature_len is {args.feature_len}\n'
                 f'\td_model is {args.d_model}\n'
                 f'\targs.nheads is {args.nheads}\n'
                 f'\targs.num_encoder_layers is {args.num_encoder_layers}\n'
                 f'\targs.dim_feedforward is {args.dim_feedforward}\n'
                 f'\targs.dropout is {args.dropout}\n'
                 f'\targs.activation is {args.activation}\n')
    if args.load_rt_param_dir:
        model.load_state_dict(torch.load(args.load_rt_param_dir, map_location=device))
        logging.info(f'Model parameters loaded from {args.load_rt_param_dir}')
    model.to(device=device)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    train(model=model,
          device=device,
          epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.lr,
          val_percent=args.vali_rate,
          sch = args.sch,
          traindir=traindir,
          validir=validir,
          checkpoint_dir=checkpoint_dir,
          expect_performance=args.expect_performance)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s') 
    args = get_args()  ##生成参数列表
    do_train(args)
