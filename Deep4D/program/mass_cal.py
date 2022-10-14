########用于从原始二级谱图中识别指定肽段的b,y离子
import numpy as np
import pandas as pd
from constant import Mass
import math

def regular_peptide_msms_m_z(peptide, by_type, num, charge, loss):  #####计算普通肽的二级b，y离子的m/z，
    a = 0
    if peptide[0] == 'a':
        a = 1
        peptide = peptide.replace('a','')
    if by_type == 'b':
        b = peptide
    elif by_type == 'y':
        b = peptide[::-1]
    mass = 0
    for j in range(num):
        mass = mass + Mass.AA_residue_mass[b[j]]
    mass = mass - Mass.loss_mass[loss]
    if by_type == 'b':
        if a == 1:
            mass = mass + Mass.AA_residue_mass['a']
        mass = mass + 1.00783 ###加上氢原子
        m_z = mass / charge + (charge - 1)*1.00728/charge  ###因为碎裂后C原子带正点，所以对于带N个正电的时候，只需要加N-1个H+
    elif by_type == 'y':
        mass = mass + 17.00274  ###加上羟基
        m_z = mass / charge + (charge + 1)*1.00728/charge   ###碎裂后带有原来肽的H+，但是和N上的负电中和了，所以要带N个正点的话，需要加N+1个H+
    return m_z

def regular_peptide_m_z(peptide, charge):
    mass = 0
    for i in range(len(peptide)):
        mass = mass + Mass.AA_residue_mass[peptide[i]]
    mass = mass + Mass.loss_mass['H2O']
    m_z = mass/charge + Mass.loss_mass['H+']
    return m_z

def calculate_ccs(peptide_m_z, peptide_charge, peptide_k0):
    m = 28.00615
    t = 304.7527
    coeff = 18500 * peptide_charge * math.sqrt((peptide_m_z * peptide_charge + m) / (peptide_m_z * peptide_charge * m * t))
    ccs = coeff*peptide_k0
    return ccs

def calculate_k0(peptide_m_z, peptide_charge, peptide_ccs):
    m = 28.00615
    t = 304.7527
    coeff = 18500 * peptide_charge * math.sqrt((peptide_m_z * peptide_charge + m) / (peptide_m_z * peptide_charge * m * t))
    k0 = peptide_ccs/coeff
    return k0

if __name__ == '__main__':
    # print(regular_peptide_msms_m_z('AAAAAAAAAPAAAATAPTTAATTAATAAQ','b',27,2,'NH3'))
    print(regular_peptide_m_z('AVFVDIEPTVIDEVR',2))


