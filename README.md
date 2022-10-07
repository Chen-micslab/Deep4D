# Deep4D
## Introduction
Deep4D is a deep learning model which could predict ion mobility, retention time, fragment intensity and charge state of peptide to generate in silico 4D library for 4D DIA proteomics and phosphoproteomics.
If necessary, Deep4D can also provide separate ion mobility, retention time, fragment intensity or charge state prediction functions.
## Guide to generate a 4D library 
### 1. Prepare the peptide list
The peptide list should be stored in a comma-separated values (CSV) file including two column:'Peptide','Charge'. This CSV file should be stored at the directory 'Deep4D/Deep4D/dataset/data'
```
Peptide,Charge
AAAAAAAAGAFAGR,2
aAAAAAAAAVPSAAGR,2
AAAAAATAPPSPGPAQPGPR,2
AAAAALeSQQQSLQER,2
AAAAAWEEPSSGNGTAR,2
AAAAFVLsANENNIALFK,2
```
The PTM contain phosphorylation of serine, threonine and tyrosine, oxidation of methionine and acetylation on N-terminal of proteins. They are represented as: S(phos)--s, T(phos)--t, Y(phos)--y, M(oxid)--e, acetylation--a.

### 2. Generate 4D library
The generation of 4D library is executed through this script 'Deep4D/Deep4D/predict_4d_library.py'. The parameters of ccs model, rt model and msms model are required and should be stored as 'pth' file at the directory 'Deep4D/Deep4D/checkpoint'.   
Run `'Deep4D/Deep4D/predict_4d_library.py'`.  
Example:
```
python predict_4d_library.py --filename 'peptide_list.csv' --load_msms_param_dir './checkpoint/msms_model.pth' --load_ccs_param_dir './checkpoint/ccs_model.pth' --load_rt_param_dir './checkpoint/rt_model.pth' --maxcharge 3 --type 'DeepDIA' --library 'Peaks' --slice 3 --batch_size 50 
```
Description of argparse:  
--filename: the 'csv' file of peptide list  
--load_msms_param_dir: the file directory of msms model  
--load_ccs_param_dir: the file directory of ccs model  
--load_rt_param_dir: the file directory of rt model  
--maxcharge: the maximum charge in peptide list, the charge range would in [2,maxcharge], the recommended range is [2,4]  
--type: The neutral loss type of msms spectrum. Now Deep4D contains two type 'DeepDIA' and 'DeepPhospho'. 'DeepDIA': NH3 and H20, 'DeepPhospho': H3PO4.  
--library: The type of 4D library. Now Deep4D contains only 'Peaks' which support PEAKS Online software.    
--slice: If the scale of  peptide list is too large for your compute, you can slice the list. It should be set to an integer greater than 0.  
--batch_size: The batch size.  
