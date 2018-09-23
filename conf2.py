import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
#R:577
#D:377
network_path = '../IMCdata/'

# DS semantic:DS1.txt(DS1mask.txt),DS2.txt(DS2mask.txt);
# DS functional:DSfunctional.txt ;
# DS gaussian:DSgaussian.txt;
DS1 = np.loadtxt(network_path+'DS/DS1.txt')
DS2 = np.loadtxt(network_path+'DS/DS2.txt')
DSfc=np.loadtxt(network_path+'DS/DSfunctional.txt')
DSgs=np.loadtxt(network_path+'DS/DSgaussian.txt')

DSmask1 = np.loadtxt(network_path+'DS/DSmask1.txt')
DSmask2= np.loadtxt(network_path+'DS/DSmask2.txt')

# RS functional:RSfunctional.txt ;
# RS gaussian:RSgaussian.txt;
# RS sequence:RSsequence.txt;
RSfc = np.loadtxt(network_path+'RS/RSfunctional.txt')
RSgs=np.loadtxt(network_path+'RS/RSgaussian.txt')
RSsq=np.loadtxt(network_path+'RS/RSsequence.txt')

RSmask = np.loadtxt(network_path+'RS/RSfunctionalmask.txt')


num_R = len(RSfc)
num_D = len(DS1)
DR_sparse=np.loadtxt(network_path+'DR.txt')
RD = coo_matrix((np.array([1]*DR_sparse.shape[0]), (DR_sparse[:, 0], DR_sparse[:, 1])),shape=(num_R,num_D)).todense()
DR=RD.T


dim_R=500
dim_D=500
dim_pred = 512
dim_pass = 500
GN=0.2  #global norm to be clipped
PNR='all'  # positive negative ratio
testS='o'  #Test scenario

diseasedict={49:'Breast neoplasms',58:'Hepatocellular carcinoma',62:'Renal cell carcinoma',64:'Squamous cell carcinoma',
      92:'Colorectal neoplasms',144:'Glioblastoma',162:'Heart failure',218:'Acute myeloid leukemia',235:'Lung neoplasms',
      252:'Melanoma',303:'Ovarian neoplasms',306:'Pancreatic neoplasms',327:'Prostatic neoplasms',363:'Stomach neoplasms',
            376:'Urinary bladder neoplasms'}
#50,59,63,65,93,145,163,219,236,253,304,307,328,364,377