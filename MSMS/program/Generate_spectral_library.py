import numpy as np
import pandas as pd

def generate_4d_library(data, type): #####data是predict_npy输出的numpy矩阵, type有DeepDIA和DeepPhospho
    peptide_list = []
    charge_list = []
    m_z_list = []
    rt_list = []
    ccs_list = []
    ion_charge = []
    ion_type = []
    ion_num = []
    ion_loss = []
    ion_inten = []
    if type == 'DeepDIA':
        peptide = data[:,0]
        charge = data[:,1]
        pep_m_z = data[:,2]
        pep_len = data[:,3]
        rt = data[:,-1]
        ccs = data[:,-2]
        pre_msms = data[:,4:-2]
        for j in range(len(peptide)):
            for i in range(pep_len[j]*12):
                if pre_msms[j][i] > 0:
                    a = i % 12
                    num = (i//12) + 1
                    peptide_list.append(peptide[j])
                    charge_list.append(charge[j])
                    m_z_list.append(pep_m_z[j])
                    rt_list.append(rt[j])
                    ccs_list.append(ccs[j])
                    ion_num.append(num)
                    ion_inten.append(pre_msms[j][i])
                    if a == 0:
                        ion_charge.append(1)
                        ion_type.append('b')
                        ion_loss.append('Noloss')
                    elif a == 1:
                        ion_charge.append(1)
                        ion_type.append('b')
                        ion_loss.append('NH3')
                    elif a == 2:
                        ion_charge.append(1)
                        ion_type.append('b')
                        ion_loss.append('H2O')
                    elif a == 3:
                        ion_charge.append(2)
                        ion_type.append('b')
                        ion_loss.append('Noloss')
                    elif a == 4:
                        ion_charge.append(2)
                        ion_type.append('b')
                        ion_loss.append('NH3')
                    elif a == 5:
                        ion_charge.append(2)
                        ion_type.append('b')
                        ion_loss.append('H2O')
                    elif a == 6:
                        ion_charge.append(1)
                        ion_type.append('y')
                        ion_loss.append('Noloss')
                    elif a == 7:
                        ion_charge.append(1)
                        ion_type.append('y')
                        ion_loss.append('NH3')
                    elif a == 8:
                        ion_charge.append(1)
                        ion_type.append('y')
                        ion_loss.append('H2O')
                    elif a == 9:
                        ion_charge.append(2)
                        ion_type.append('y')
                        ion_loss.append('Noloss')
                    elif a == 10:
                        ion_charge.append(2)
                        ion_type.append('y')
                        ion_loss.append('NH3')
                    elif a == 11:
                        ion_charge.append(2)
                        ion_type.append('y')
                        ion_loss.append('H2O')
    elif type == 'DeepPhospho':
        peptide = data[:,0]
        charge = data[:,1]
        pep_m_z = data[:,2]
        pep_len = data[:,3]
        rt = data[:,-1]
        ccs = data[:,-2]
        pre_msms = data[:,4:-2]
        for j in range(len(peptide)):
            for i in range(pep_len[j]*8):
                if pre_msms[j][i] > 0:
                    a = i % 8
                    num = (i//8) + 1
                    peptide_list.append(peptide[j])
                    charge_list.append(charge[j])
                    m_z_list.append(pep_m_z[j])
                    rt_list.append(rt[j])
                    ccs_list.append(ccs[j])
                    ion_num.append(num)
                    ion_inten.append(pre_msms[j][i])
                    if a == 0:
                        ion_charge.append(1)
                        ion_type.append('b')
                        ion_loss.append('Noloss')
                    elif a == 1:
                        ion_charge.append(1)
                        ion_type.append('b')
                        ion_loss.append('H3PO4')
                    elif a == 2:
                        ion_charge.append(2)
                        ion_type.append('b')
                        ion_loss.append('Noloss')
                    elif a == 3:
                        ion_charge.append(2)
                        ion_type.append('b')
                        ion_loss.append('H3PO4')
                    elif a == 4:
                        ion_charge.append(1)
                        ion_type.append('y')
                        ion_loss.append('Noloss')
                    elif a == 5:
                        ion_charge.append(1)
                        ion_type.append('y')
                        ion_loss.append('H3PO4')
                    elif a == 6:
                        ion_charge.append(2)
                        ion_type.append('y')
                        ion_loss.append('Noloss')
                    elif a == 7:
                        ion_charge.append(2)
                        ion_type.append('y')
                        ion_loss.append('H3PO4')
    else:
        print(f'type {type} does not exist!')
    data1 = pd.DataFrame()
    data1['Peptide'] = peptide_list
    data1['Charge'] = charge_list
    data1['m_z'] = m_z_list
    data1['RT'] = rt_list
    data1['CCS'] = ccs_list
    data1['FI.Charge'] = ion_charge
    data1['FI.FrgType'] = ion_type
    data1['FI.FrgNum'] = ion_num
    data1['FI.LossType'] = ion_loss
    data1['FI.Intensity'] = ion_inten
    return data1