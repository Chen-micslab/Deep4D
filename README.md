# Deep4D
## Introduction
Deep4D is a deep learning model which could predict ion mobility, retention time, fragment intensity and charge state of peptide to generate in silico 4D library for 4D DIA proteomics and phosphoproteomics.
If necessary, Deep4D can also provide separate ion mobility, retention time, fragment intensity or charge state prediction functions.

If you have any questions, you can send feedback to moranchen111@gmail.com.
## Publications
M. Chen, P. Zhu, Q. Wan, X. Ruan, P. Wu, Y. Hao, Z. Zhang, J. Sun, W. Nie, S. Chen*. High-Coverage Four-Dimensional Data-Independent Acquisition Proteomics and Phosphoproteomics Enabled by Deep Learning-Driven Multidimensional Predictions. Anal. Chem. 2023, 95, 7495–7502. https://pubs.acs.org/doi/10.1021/acs.analchem.2c05414
## Guide to generate a 4D library 
### 1. Prepare peptide list
The peptide list should be stored in a comma-separated values (CSV) file including two column:'Peptide', 'Charge'. This CSV file should be stored at the directory 'Deep4D/dataset/data/peptide_list.csv'
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
The generation of 4D library is executed through this script 'Deep4D/predict_4d_library.py'. The parameters of ccs model, rt model and msms model are required and should be stored as 'pth' file at the directory 'Deep4D/checkpoint'.   
Run `'Deep4D/predict_4d_library.py'`.  
Example:
```
python predict_4d_library.py --filename 'peptide_list.csv' --load_msms_param_dir './checkpoint/msms_model.pth' --load_ccs_param_dir './checkpoint/ccs_model.pth' --load_rt_param_dir './checkpoint/rt_model.pth' --maxcharge 3 --type 'DeepDIA' --library 'Peaks' --slice 3 --batch_size 50 
```
#### Description of argparse:  
--filename: the 'csv' file of peptide list  
--load_msms_param_dir: the file directory of msms model  
--load_ccs_param_dir: the file directory of ccs model  
--load_rt_param_dir: the file directory of rt model  
--maxcharge: the maximum charge in peptide list, the charge range would in [2,maxcharge], the recommended range is [2,4]  
--type: The neutral loss type of msms spectrum. Now Deep4D contains two type 'DeepDIA' and 'DeepPhospho'. 'DeepDIA': NH3 and H20, 'DeepPhospho': H3PO4.  
--library: The type of 4D library. Now Deep4D contains the normal library which support DIA-NNsoftware and 'Peaks' which support PEAKS Online software.    
--slice: If the scale of  peptide list is too large for your compute, you can slice the list. It should be set to an integer greater than 0.  
--batch_size: The batch size.  
#### 4D Library formats
The formats generated by Deep4D now contain two type: 'Peaks' and 'Normal'. 'Peaks': tab-separated (.tsv), 'Normal': comma-separated (.csv).   
1. 'Normal': It contains all the necessary information of a conventional spectrum library. The columns:
- **Peptide**: Modified peptide
- **Charge**: Charge state of peptide
- **m_z**: m/z of peptide
- **RT**: Retention time
- **IM**: Ion mobility
- **FI.Charge**: Charge state of fragment 
- **FI.FrgType**: b or y type of fragment
- **FI.FrgNum**: Series number of fragment
- **FI.LossType**: Neutral loss typep of fragment
- **FI.Intensity**: Intensity of fragment
- **FI.m_z**: m/z of fragment
## Guide to train the model
### a. Train ccs model  
#### 1. Prepare the training data
The training data should be stored in a comma-separated values (CSV) file including three columns:'Peptide', 'Charge', 'CCS'. This CSV file should be stored at the directory 'CCS/dataset/data/ccs_data.csv'.
```
Peptide,Charge,CCS
aAADIASK,1,218.8175142
aAAQGEPQVQFK,1,264.0467586
GHFNIQPNKK,3,433.9747944
IAILGFAFKK,2,382.8162064
KIGEEEIQKPEEK,2,415.801214
LQAEIKR,2,323.5481144
```
#### 2. Encoding peptide
Run `'CCS/dataset/Encoding_ccs.py'`. 
```
python Encoding_ccs.py --filename 'ccs_data' --label 1
```
--label: If label ccs exist, label = 1. If no label ccs exist, label = 0.
#### 3. Train ccs model
Run `'CCS/train_ccs.py'` for single charge state or `'CCS/train_ccs_z.py'` for multiple charge states. 
```
python train_ccs_z.py --filename 'ccs_data' --load_ccs_param_dir './checkpoint/ccs.pth' --lr 0.00001 --batch_size 50
```
--filename: Training data name.  
--load_ccs_param_dir: The file directory of pre-trained ccs model.  
--lr: Learning rate.  
Finally, find the parameters file at './checkpoint/{ccs_data}_ccs/'
#### 4. Predict ccs
Run `'CCS/predict_ccs.py'` for single charge state or `'CCS/predict_ccs_z.py'` for multiple charge states.  
```
python train_ccs_z.py --filename 'ccs_test_data' --load_ccs_param_dir './checkpoint/ccs.pth' --batch_size 50 --label 1
```
--filename: Test data name, which was also encoded with step 2.  
--label: If label ccs exist, label = 1. If no label ccs exist, label = 0.
### b. Train rt model  
#### 1. Prepare the training data
The training data should be stored in a comma-separated values (CSV) file including two columns:'Peptide', 'RT'. This CSV file should be stored at the directory 'RT/dataset/data/rt_data.csv'.
```
Peptide,RT
aAADIASK,6.141666667
aAAQGEPQVQFK,34.835
GHFNIQPNKK,12.24
IAILGFAFKK,51.305
KIGEEEIQKPEEK,16.84666667
LQAEIKR,12.26
```
#### 2. Encoding peptide
Run `'RT/dataset/Encoding_rt.py'`. 
```
python Encoding_rt.py --filename 'rt_data' --label 1
```
--label: If label rt exist, label = 1. If no label rt exist, label = 0.
#### 3. Train rt model
Run `'RT/train_rt.py'`. 
```
python train_rt.py --filename 'rt_data' --load_rt_param_dir './checkpoint/rt.pth' --lr 0.00001 --batch_size 50
```
--filename: Training data name.  
--load_rt_param_dir: The file directory of pre-trained rt model.  
--lr: Learning rate.  
Finally, find the parameters file at './checkpoint/{rt_data}_rt/'
#### 4. Predict rt
Run `'RT/predict_rt.py'`. 
```
python predict_rt.py  --filename 'rt_test_data' --load_rt_param_dir './checkpoint/rt.pth' --batch_size 50 --label 1
```
--filename: Test data name, which was also encoded with step 2.  
--label: If label rt exist, label = 1. If no label rt exist, label = 0.
### c. Train msms model  
#### 1. Prepare the training data
The training data should be stored in a comma-separated values (CSV) file including seven columns: 'Peptide', 'Charge', 'FI.Charge', 'FI.FrgType', 'FI.FrgNum', 'FI.LossType', 'FI.Intensity'. This CSV file should be stored at the directory 'MSMS/dataset/data/msms_data.csv'.
```
Peptide,Charge,FI.Charge,FI.FrgType,FI.FrgNum,FI.LossType,FI.Intensity
aAAAGSAAVsGAGtPVAGPTGR,2,1,b,4,noloss,0.1500
aAAAGSAAVsGAGtPVAGPTGR,2,1,b,5,noloss,0.1417
aAAAGSAAVsGAGtPVAGPTGR,2,1,y,4,noloss,0.2713
aAAAGSAAVsGAGtPVAGPTGR,2,1,b,6,noloss,0.3218
aAAAGSAAVsGAGtPVAGPTGR,2,1,y,5,noloss,1.0000
aAAAGSAAVsGAGtPVAGPTGR,2,1,b,7,noloss,0.4741
aAAAGSAAVsGAGtPVAGPTGR,2,1,y,6,noloss,0.8539
```
#### 2. Encoding peptide
Run `'MSMS/dataset/Encoding_msms.py'`. 
```
python Encoding_msms.py --filename 'msms_data' --label 1 --type 'DeepDIA' --maxcharge 3
```
--label: If label msms exist, label = 1. If no label msms exist, label = 0.
--type: The neutral loss type of msms spectrum. Now Deep4D contains two type 'DeepDIA' and 'DeepPhospho'. 'DeepDIA': NH3 and H20, 'DeepPhospho': H3PO4.
--maxcharge: the maximum charge in peptide list, the charge range would in [2,maxcharge], the recommended range is [2,4]  
#### 3. Train msms model
Run `'MSMS/train_msms.py'`. 
```
python train_msms.py --filename 'msms_data' --load_msms_param_dir './checkpoint/msms.pth' --lr 0.0001 --batch_size 50  --type 'DeepDIA'
```
--filename: Training data name.  
--load_msms_param_dir: The file directory of pre-trained msms model.  
--lr: Learning rate.  
--type: The neutral loss type of msms spectrum. Now Deep4D contains two type 'DeepDIA' and 'DeepPhospho'. 'DeepDIA': NH3 and H20, 'DeepPhospho': H3PO4.  
Finally, find the parameters file at './checkpoint/{msms_data}_{type}/'
#### 4. Predict msms
Run `'MSMS/predict_msms.py'`. 
```
python predict_msms.py  --filename 'msms_data' --load_msms_param_dir './checkpoint/msms.pth' --batch_size 50  --type 'DeepDIA' --label 1
```
--filename: Test data name, which was also encoded with step 2.  
--label: If label msms exist, label = 1. If no label msms exist, label = 0.
### d. Train charge state model  
#### 1. Prepare the training data
The training data should be stored in a comma-separated values (CSV) file including seven columns: 'Peptide', 'Charge'. This CSV file should be stored at the directory 'Charge_state/dataset/data/charge_data.csv'.  
```
Peptide,Charge
AAAAAAAAGAFAGR,2
aAAAAAAAAVPSAAGR,5
AAAAAATAPPSPGPAQPGPR,7
AAAAALeSQQQSLQER,9
AAAAAWEEPSSGNGTAR,3
AAAAFVLsANENNIALFK,4
```
Here, the charge of each peptide is the sum of all its charge states. Example: a peptide with charge state 2 and 3, so its charge = 5.
#### 2. Encoding peptide
Run `'Charge_state/dataset/Encoding_charge.py'`. 
```
python Encoding_charge.py --filename 'charge_data' --label 1 
```
--label: If label msms exist, label = 1. If no label msms exist, label = 0.
#### 3. Train charge model
Run `'Charge_state/train_charge.py'`. 
```
python train_charge.py --filename 'charge_data' --load_param_dir './checkpoint/charge.pth' --lr 0.00001 --batch_size 50
```
--filename: Training data name.  
--load_param_dir: The file directory of pre-trained charge model.  
--lr: Learning rate.  
Finally, find the parameters file at './checkpoint/{charge_data}_charge/'
#### 4. Predict charge
Run `'Charge_state/predict_rt.py'`. 
```
python predict_charge.py  --filename 'charge_test_data' --load_param_dir './checkpoint/charge.pth' --batch_size 50 
```
--filename: Test data name, which was also encoded with step 2. 
## License
Deep4D is distributed under an Apache License. See the LICENSE file for details.
## Contacts
Please report any problems directly to the github issue tracker. Also, you can send feedback to moranchen111@gmail.com.
