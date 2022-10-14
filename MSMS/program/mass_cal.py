import numpy as np
import pandas as pd
from constant import Mass

def regular_peptide_msms_m_z(peptide, by_type, num, charge, loss):  
    if by_type == 'b':
        b = peptide
    elif by_type == 'y':
        b = peptide[::-1]
    mass = 0
    for j in range(num):
        mass = mass + Mass.AA_residue_mass[b[j]]
    mass = mass - Mass.loss_mass[loss]
    if by_type == 'b':
        mass = mass + 1.00783 ###加上氢原子
        m_z = mass / charge + (charge - 1)*1.00728/charge  
    elif by_type == 'y':
        mass = mass + 17.00274  ###加上羟基
        m_z = mass / charge + (charge + 1)*1.00728/charge   
    return m_z

def regular_peptide_m_z(peptide, charge):
    mass = 0
    for i in range(len(peptide)):
        mass = mass + Mass.AA_residue_mass[peptide[i]]
    mass = mass + Mass.loss_mass['H2O']
    m_z = mass/charge + Mass.loss_mass['H+']
    return m_z




