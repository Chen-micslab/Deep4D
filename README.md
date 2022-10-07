# Deep4D
## Introduction
Deep4D is a deep learning model which could predict ion mobility, retention time, fragment intensity and charge state of peptide to generate in silico 4D library for 4D DIA proteomics and phosphoproteomics.
If necessary, Deep4D can also provide separate ion mobility, retention time, fragment intensity or charge state prediction functions
## Guide to generate a 4D library 
### 1. Prepare the peptide list
The peptide list should be stored in a comma-separated values (CSV) file including two column:'Peptide','Charge'. 
```
Peptide,Charge
AAAAAAAAGAFAGR,2
AAAAAAAAVPSAGPAGPAPTSAAGR,2
AAAAAATAPPSPGPAQPGPR,2
AAAAALSQQQSLQER,2
AAAAAWEEPSSGNGTAR,2
AAAAFVLANENNIALFK,2
```
The PTM contain phosphorylation of serine, threonine and tyrosine, oxidation of methionine and acetylation on N-terminal of proteins.
