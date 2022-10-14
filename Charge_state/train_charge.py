import argparse
import  logging
import os
import sys

import numpy as  np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from model import selfatt_cnn_charge
from dataset.Dataset_charge import Mydata_label
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.Eval_model_charge import eval_model
from model.selfatt_cnn_charge import Transformer


####################用于测试不同训练集数量对于效果的影响


def get_args():  ##设置需要传入的参数
    parser = argparse.ArgumentParser(description='Train charge state model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--feature_len', type=int, default=23)
    parser.add_argument('--d_model', type=int, default=500)
    parser.add_argument('--nheads', type=int, default=5)
    parser.add_argument('--num_encoder_layers', type=int, default=5)
    parser.add_argument('--dim_feedforward', type=int, default=1200)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--norm', type=float, default=1)
    parser.add_argument('--validation', type=float, default=0.1) ##validation占总训练数据的比例
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--output_dir', type=str, default='./output')  ##
    parser.add_argument('--load_param_dir', type=str, default=None) ##定义load模型参数的文件路径，默认为false
    parser.add_argument('--seed', type=int, default=1)  ##定义load模型参数的文件路径，默认为false
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--sch', type=int, default=0)
    parser.add_argument('--train', type=float, default=1)
    return parser.parse_args()

def get_lr(epoch):  ##定义lr的调整规则
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
        mask[i, :int(length[i])] = 1
    return  mask

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

###设置模型训练时的细节
def train(model,
          device,
          epochs=10,
          batch_size=1,
          lr=0.001,
          val_percent=0.1,
          train_percent = 0.1,
          save_mp=True,  ##是否保存模型的参数
          sch = 0
          ):

    mydata = Mydata_label(data_dir)  ##导入dataset
    n_val = int(len(mydata) * val_percent)  ##计算validation data的数量
    n_train = len(mydata) - n_val   ##计算train data的数量
    train_data, val_data = random_split(mydata ,[n_train, n_val])  ##随机分配validation data和train data
    n_train_ac = int(len(train_data) * train_percent)  #使用所有训练集中的多少数据作为真正的训练集
    train_ac, train_remain = random_split(train_data,[n_train_ac,(len(train_data)-n_train_ac)]) #从训练集中按照设置的比例划分出真正的训练集
    train_loader = DataLoader(train_ac,batch_size=batch_size,shuffle=True,num_workers=3,pin_memory=True)  ##生成train data
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True, drop_last=True) ##生成validation data
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
    optimizer =optim.Adam( model.parameters(), lr=lr, betas=(0.9,0.98),eps=1e-09, weight_decay=0, amsgrad=False) ##选择optimizer为Adam,注意传入的是model的parameters
    if sch==0:
         scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=4,T_mult=2,eta_min=0.00000001)
    else:
         scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=get_lr)
    best_index = 0
    for epoch in range(epochs):
        model.train()  ##设置模型为train模式
        local_step = 0
        epoch_loss = 0
        with tqdm(total=n_train_ac, desc=f'Epoch {epoch+1}/{epochs}',unit='peptide') as pbar:
            for batch in train_loader:
                local_step += 1
                peptide = batch['peptide_onehot'].to(device=device, dtype=torch.float32)
                length = batch['length'].to(device=device, dtype=torch.float32)
                charge = batch['charge'].to(device=device,dtype=torch.float32)
                charge = transfer_charge(charge,device)
                mask = get_mask(peptide,length).to(device=device,dtype=torch.bool)
                charge_pred = model(src=peptide,src_key_padding_mask=mask)
                loss_f = nn.BCELoss()
                loss = loss_f(charge_pred,charge)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step=global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()  ##梯度清零
                loss.backward()  ##反向传播
                optimizer.step()
                pbar.update(peptide.shape[0])
                global_step += 1
                if global_step % (n_train_ac // (2 * batch_size)) == 0:
                    val_acc = eval_model(model,val_loader,device)
                    logging.info('Validation Accuracy: {}'.format(val_acc))
                    if val_acc > best_index:
                        torch.save(model.state_dict(),
                                   checkpoint_dir + f'model_param_epoch{epoch + 1}global_step{global_step}val_acc{val_acc}.pth')  ###只保存模型的参数到checkpoint_dir
                        logging.info(f'Checkpoint {epoch + 1}global_step{global_step} saved !')
        scheduler.step()
        if save_mp:
            logging.info('Created checkpoint directory')
            torch.save(model.state_dict(), checkpoint_dir + f'model_param_epoch{epoch + 1}train{train_percent}.pth')###只保存模型的参数到checkpoint_dir
            logging.info(f'Checkpoint {epoch + 1} saved !')
    writer.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    data_dir = f'./dataset/data/{args.filename}' ##训练数据的路径
    checkpoint_dir = f'./checkpoint/{args.filename}_charge/'  ##checkpoint文件夹的路径，用于保存模型参数
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
          train_percent = args.train,
          save_mp=True,
          sch = args.sch
              )

