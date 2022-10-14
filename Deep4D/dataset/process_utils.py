import numpy as np
import pandas as pd

def get_index(peptide, *chars):
    loc = []
    for char in chars:
        if char in peptide:
            loc.append(peptide.find(char))
            loc.append(peptide.rfind(char))
    loc = set(loc)
    return loc

def filter_phos_ion(peptide, pep_matrix):   ####把不可能出现H3P04中性丢失的碎片变为-1
    peptide = peptide.replace('a', '')
    pep_len = len(peptide)
    phos_index = list(get_index(peptide,'s','t','y'))
    if len(phos_index) == 0:
        pass
    elif len(phos_index) == 1:
        min_id = phos_index[0]
        max_id = pep_len - phos_index[0] - 1
        if min_id == 0:
            pass
        else:
            for i in range(min_id):
                pep_matrix[i, 1] = -1
                pep_matrix[i, 3] = -1
        if max_id == 0:
            pass
        else:
            for i in range(max_id):
                pep_matrix[i, 5] = -1
                pep_matrix[i, 7] = -1
    elif len(phos_index) > 1:
        min_id = min(phos_index)
        max_id = pep_len - max(phos_index) - 1
        if min_id == 0:
            pass
        else:
            for i in range(min_id):
                pep_matrix[i, 1] = -1
                pep_matrix[i, 3] = -1
        if max_id == 0:
            pass
        else:
            for i in range(max_id):
                pep_matrix[i, 5] = -1
                pep_matrix[i, 7] = -1
    return pep_matrix

