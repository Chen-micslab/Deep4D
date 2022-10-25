import numpy as np
import pandas as pd
from mass_cal import calculate_k0, regular_peptide_msms_m_z


def merge_information(msms_dir, ccs_dir, rt_dir):
    msms_matrix = np.array(pd.read_csv(msms_dir))
    ccs_matrix = np.array(pd.read_csv(ccs_dir))
    rt_matrix = np.array(pd.read_csv(rt_dir))
    m_z = msms_matrix[:,2]
    charge = msms_matrix[:,1]
    ccs = ccs_matrix[:,2]
    rt = rt_matrix[:,2]
    k0 = []
    for i in range(len(ccs)):
        k0.append(calculate_k0(m_z[i], charge[i], ccs[i]))
    k0 = np.array(k0)
    rt = np.expand_dims(np.array(rt), axis=1)
    k0 = np.expand_dims(np.array(k0), axis=1)
    data = np.column_stack((msms_matrix, k0, rt))
    return data

def generate_4d_library(data, type): #####type有DeepDIA和DeepPhospho
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
            if len(pre_msms[j,:][(pre_msms[j,:] > 0)]) > 2: ####过滤碎片数小于3的peptide
                for i in range(pep_len[j]*12):
                    if pre_msms[j][i] >= 0.02:  ###过滤强度小于最高峰2%的碎片
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
            if len(pre_msms[j,:][(pre_msms[j,:] > 0)]) > 2: ####过滤碎片数小于3的peptide
                for i in range(pep_len[j]*8):
                    if pre_msms[j][i] >= 0.01:  ###过滤强度小于最高峰5%的碎片
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
    data1['IM'] = ccs_list
    data1['FI.Charge'] = ion_charge
    data1['FI.FrgType'] = ion_type
    data1['FI.FrgNum'] = ion_num
    data1['FI.LossType'] = ion_loss
    data1['FI.Intensity'] = ion_inten
    data = np.array(data1)
    msms_mz = data[:, 2].copy()
    for i in range(len(data[:, 0])):
        msms_mz[i] = regular_peptide_msms_m_z(data[i, 0], data[i, 6], data[i, 7], data[i, 5], data[i, 8])
    msms_mz = np.array(msms_mz, dtype=float).round(5)
    data1['FI.m_z'] = list(msms_mz)
    return data1

def move_modification(peptide):
    peptide = peptide.replace('a', '')
    peptide = peptide.replace('e', 'M')
    peptide = peptide.replace('s', 'S')
    peptide = peptide.replace('t', 'T')
    peptide = peptide.replace('y', 'Y')
    return peptide

class change_to_peaks():
    def __init__(self):
        self.modification = {'a':'0-Acetylation (N-term);',
                             'e':'-Oxidation (M);',
                             's':'-Phosphorylation (STY);',
                             't':'-Phosphorylation (STY);',
                             'y':'-Phosphorylation (STY);',
                             'C':'-Carbamidomethylation;'}
        self.loss = {'NH3':'-NH3', 'H2O':'-H2O', 'Noloss':'','noloss':''}
        self.min_peaks = 3

    def forward(self, data):
        charge_list = list(set(data['Charge']))  ###eg: maxcharge = 4, charge_list = [2, 3, 4]
        name = ['m/z','z','rt (seconds)','TimsTOF 1/k0','Activation Mode','Sequence (backbone)','Modifications','Peaks Count','Peaks List']
        m_z_list, z_list, rt_list, im_list, mode_list, pep_list, modif_list, peaks_count_list, peak_list = [], [], [], [], [], [], [], [], []
        for z in charge_list:
            data1 = data[data['Charge'] == z]
            if len(np.array(data1['Peptide'])) > 0:
                data1 = data1.sort_values('Peptide', ignore_index=True)  ####按照肽名字排序
                peptide_list = np.array(data1['Peptide'])
                peptide = peptide_list[0]
                index_list = []
                for i in range(len(peptide_list)):
                    if peptide_list[i] == peptide:
                        index_list.append(i)
                    else:
                        data2 = data1.iloc[index_list]
                        data2 = self.choose_top_n(data2, 20)
                        charge = np.array(data2['Charge'])[0]
                        m_z = round(np.array(data2['m_z'])[0],5)
                        rt = round(np.array(data2['RT'])[0] * 60,1)
                        im = round(np.array(data2['IM'])[0],4)
                        ion_charge = np.array(data2['FI.Charge'])
                        ion_type = np.array(data2['FI.FrgType'])
                        ion_num = np.array(data2['FI.FrgNum'])
                        ion_loss = np.array(data2['FI.LossType'])
                        ion_inten = np.array(data2['FI.Intensity'])
                        ion_m_z = np.array(data2['FI.m_z'])
                        msms, peaks_num = self.integrate_MSMS(ion_inten, ion_m_z, ion_charge, ion_type, ion_num, ion_loss, charge)
                        modif = self.extract_peaks_modif(peptide)
                        norm_pep = self.modpep_to_normpep(peptide)
                        if peaks_num > self.min_peaks:
                            m_z_list.append(m_z)
                            z_list.append(charge)
                            rt_list.append(rt)
                            im_list.append(im)
                            mode_list.append('CID, CAD(y and b ions)')
                            pep_list.append(norm_pep)
                            modif_list.append(modif)
                            peaks_count_list.append(peaks_num)
                            peak_list.append(msms[:-1])
                        index_list = []
                        index_list.append(i)
                        peptide = peptide_list[i]
                data2 = data1.iloc[index_list]
                data2 = self.choose_top_n(data2, 20)
                charge = np.array(data2['Charge'])[0]
                m_z = round(np.array(data2['m_z'])[0],5)
                rt = round(np.array(data2['RT'])[0] * 60,1)
                im = round(np.array(data2['IM'])[0],4)
                ion_charge = np.array(data2['FI.Charge'])
                ion_type = np.array(data2['FI.FrgType'])
                ion_num = np.array(data2['FI.FrgNum'])
                ion_loss = np.array(data2['FI.LossType'])
                ion_inten = np.array(data2['FI.Intensity'])
                ion_m_z = np.array(data2['FI.m_z'])
                msms, peaks_num = self.integrate_MSMS(ion_inten, ion_m_z, ion_charge, ion_type, ion_num, ion_loss, charge)
                modif = self.extract_peaks_modif(peptide)
                norm_pep = self.modpep_to_normpep(peptide)
                if peaks_num > self.min_peaks:
                    m_z_list.append(m_z)
                    z_list.append(charge)
                    rt_list.append(rt)
                    im_list.append(im)
                    mode_list.append('CID, CAD(y and b ions)')
                    pep_list.append(norm_pep)
                    modif_list.append(modif)
                    peaks_count_list.append(peaks_num)
                    peak_list.append(msms[:-1])
        m_z_list = np.array(m_z_list)
        z_list = np.array(z_list)
        rt_list = np.array(rt_list)
        im_list = np.array(im_list)
        mode_list = np.array(mode_list)
        pep_list = np.array(pep_list)
        modif_list = np.array(modif_list)
        peaks_count_list = np.array(peaks_count_list)
        peak_list = np.array(peak_list)
        data2 = np.column_stack((m_z_list,z_list,rt_list,im_list,mode_list,pep_list,modif_list,peaks_count_list,peak_list))
        data2 = pd.DataFrame(data2,columns=name)
        return data2

    def integrate_MSMS(self, frag_intensity, frag_mz, frag_charge, frag_type, frag_typenum, loss, charge): 
        b = np.argsort(frag_mz)
        MSMS = ''
        num = 0
        for i in b:
            if str(loss[i]) in self.loss:
                mz = str(round(frag_mz[i], 5))
                inten = str(round(frag_intensity[i], 4))
                if frag_charge[i] > 1:
                    if charge > 2:
                        type = str(frag_type[i]) + str(frag_typenum[i]) + self.loss[str(loss[i])] + '[2+]'
                        msms = mz + ':' + inten + ':' + type
                        MSMS = MSMS + msms + ';'
                        num = num + 1
                else:
                    type = str(frag_type[i]) + str(frag_typenum[i]) + self.loss[str(loss[i])]
                    msms = mz + ':' + inten + ':' + type
                    MSMS = MSMS + msms + ';'
                    num = num + 1
        return MSMS, num
    
    def choose_top_n(self, data, n):
        name = list(data)
        data1 = np.array(data)
        if len(data1) > n:
            data2 = data1[np.argsort(-data1[:,-2])]
            data3 = data2[:n,:]
            data3 = pd.DataFrame(data3, columns=name)
            return data3
        else:
            return data
        
    def extract_peaks_modif(self, pep):    ####生成Modification的信息
        modif = ''
        num = 0
        if pep[0] == 'a':
            modif = modif + self.modification['a']
            pep = pep.replace('a', '')
        for i in range(len(pep)):
            if (pep[i] == 's') or (pep[i] == 't') or (pep[i] == 'y') or (pep[i] == 'e') or (pep[i] == 'C'):
                modif = modif + f'{i}{self.modification[pep[i]]}'
                num = num + 1
        if num > 0:
            modif = modif[:-1]
        return  modif

    def modpep_to_normpep(self, peptide):
        peptide = peptide.replace('a', '')
        peptide = peptide.replace('s', 'S')
        peptide = peptide.replace('t', 'T')
        peptide = peptide.replace('y', 'Y')
        peptide = peptide.replace('e', 'M')
        return peptide


