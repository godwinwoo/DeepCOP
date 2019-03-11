import numpy as np
from rdkit.Chem import AllChem, DataStructs
from rdkit import Chem
import csv

def load_csv(file):
    # load data
    expression = []
    with open(file, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel')
        for row in reader:
            expression.append(row)
    return expression

def get_feature_dict(file, delimiter=',', key_index=0, use_int=False):
    with open(file, "r") as csv_file:
        reader = csv.reader(csv_file, dialect='excel', delimiter=delimiter)
        next(reader)
        if use_int:
            my_dict = {}
            for row in reader:
                list = []
                for value in row[1:]:
                    list.append(int(value))
                my_dict[row[key_index]] = list
            return my_dict
        return dict((row[key_index], row[1:]) for row in reader)
i = 0
finger_dimension = 2048
molecules = []
fps = []
id = []
smiles = []
names = []

import os
path = os.path.dirname(os.path.abspath(__file__))
print(path)
drug_dict = get_feature_dict('GSE92742_Broad_LINCS_pert_info.txt', delimiter='\t', use_int=False) # change for phase 2

# rnaseq drugs
drug_dict = {}
drug_dict['Enzalutamide'] = ['','','','','','CNC(=O)C1=C(F)C=C(C=C1)N1C(=S)N(C(=O)C1(C)C)C1=CC=C(C#N)C(=C1)C(F)(F)F']
drug_dict['VPC14449'] = ['','','','','','Brc1n(-c2nc(N3CCOCC3)sc2)cc(Br)n1']
drug_dict['VPC17005'] = ['','','','','','O=C(NC=1SCCN=1)c1c2c(sc1)cccc2']

count = 0
for key in drug_dict:
    count += 1
    try:
        smiles = drug_dict[key][5]
        m = Chem.MolFromSmiles(smiles)
        molecules.append(m)
        fp = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=finger_dimension), fp)
        fps.append(fp)
        id.append(key)
    except:
        print(i, key, m)
    i += 1

header = ["mol"]
for i in range(finger_dimension):
    header.append("fps" + str(i))

fps = np.array(fps).reshape(len(fps),finger_dimension)
id = np.array(id)    
id = id.reshape(len(fps), 1)
data = np.hstack((id, fps))
header = np.array(header).reshape(1, len(header))
data_header = np.vstack((header, data))
np.savetxt("morgan_2048.csv", data_header, delimiter=",", fmt="%s")
