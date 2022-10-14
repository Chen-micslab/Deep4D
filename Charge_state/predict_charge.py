import argparse
import numpy as  np
import pandas as pd
import torch
from tqdm import tqdm
from dataset.Dataset_charge import Mydata_label, Mydata_nolabel
from torch.utils.data import DataLoader
from model.selfatt_cnn_charge import Transformer


################################这个predict的输入是CSV文件，在预测肽的数量较少时适用


def get_args():  ##设置需要传入的参数
    parser = argparse.ArgumentParser(description='Predict charge state')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--feature_len', type=int, default=23)
    parser.add_argument('--d_model', type=int, default=500)
    parser.add_argument('--nheads', type=int, default=5)
    parser.add_argument('--num_encoder_layers', type=int, default=5)
    parser.add_argument('--dim_feedforward', type=int, default=1200)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--norm', type=float, default=1)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--output_dir', type=str, default='./output')  ##
    parser.add_argument('--load_param_dir', type=str, default=None)  ##定义load模型参数的文件路径，默认为false
    parser.add_argument('--seed', type=int, default=1)  ##定义load模型参数的文件路径，默认为false
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--label', type=int, default=0)
    return parser.parse_args()


def extract(str,symbol):  ###定义一个函数来提取相同的开始符和结束符中间的字符串
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

def get_mask(peptide,length):
    mask = torch.zeros(peptide.size(0),peptide.size(1))
    for i in range(length.size(0)):
        mask[i, :int(length[i])] = 1
    return  mask

def predict_with_label(args):
    data_dir = f'./dataset/data/{args.filename}'
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ##判断使用GPU还是CPU
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
    model.eval()  ##将model调整为eval模式
    test_data = Mydata_label(data_dir)
    total_lenth = len(test_data)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=3, pin_memory=True)
    n_val = len(test_loader)  # val中的batch数量
    index = 0
    Z_pre = torch.zeros(total_lenth,3).to(device=device, dtype=torch.float32)
    with tqdm(total=n_val, desc='Prediction round', unit='batch', leave=False) as pbar:
        for batch in test_loader:
            peptide = batch['peptide_onehot'].to(device=device, dtype=torch.float32)
            length = batch['length'].to(device=device, dtype=torch.float32)
            peptide_seq = np.array(batch['peptide'])
            mask = get_mask(peptide, length).to(device=device, dtype=torch.bool)
            with torch.no_grad():  ##不生成梯度，减少运算量
                z_pre = model(src=peptide,src_key_padding_mask=mask)
                id = z_pre.size(0)
                Z_pre[index:(index+id),:] = z_pre
                index = index + id
            pbar.update()
    Z_pre = Z_pre.ge(0.5)
    Z_pre = Z_pre.to(device='cpu', dtype=torch.float32).numpy()
    Z_pre = pd.DataFrame(Z_pre, columns=['charge2', 'charge3', 'charge4'])
    Z_pre.to_csv(f'./dataset/data/output/{args.filename}_pre_z.csv',index=False)   # 保存所有的预测值和真实值到一个CSV文件

def predict_without_label(args):
    data_dir = f'./dataset/data/{args.filename}'
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  ##判断使用GPU还是CPU
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
    model.eval()  ##将model调整为eval模式
    test_data = Mydata_nolabel(data_dir)
    total_lenth = len(test_data)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=3, pin_memory=True)
    n_val = len(test_loader)  # val中的batch数量
    index = 0
    Z_pre = torch.zeros(total_lenth,3).to(device=device, dtype=torch.float32)
    with tqdm(total=n_val, desc='Prediction round', unit='batch', leave=False) as pbar:
        for batch in test_loader:
            peptide = batch['peptide_onehot'].to(device=device, dtype=torch.float32)
            length = batch['length'].to(device=device, dtype=torch.float32)
            peptide_seq = np.array(batch['peptide'])
            mask = get_mask(peptide, length).to(device=device, dtype=torch.bool)
            with torch.no_grad():
                z_pre = model(src=peptide,src_key_padding_mask=mask)
                id = z_pre.size(0)
                Z_pre[index:(index+id),:] = z_pre
                index = index + id
            pbar.update()
    Z_pre = Z_pre.ge(0.5)
    Z_pre = Z_pre.to(device='cpu', dtype=torch.float32).numpy()
    Z_pre = pd.DataFrame(Z_pre, columns=['charge2', 'charge3', 'charge4'])
    Z_pre.to_csv(f'./dataset/data/output/{args.filename}_pre_z.csv',index=False)   # 保存所有的预测值和真实值到一个CSV文件

if __name__ == '__main__':
    args = get_args()
    if args.label == 1:
        predict_with_label(args)
    else:
        predict_without_label(args)