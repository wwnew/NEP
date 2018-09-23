import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
#R:577
#D:377
network_path = '../data2/'
RS = np.loadtxt(network_path+'RSmat.txt',delimiter=",")
DS = np.loadtxt(network_path+'DSmat.txt',delimiter=",")
RSmask = np.loadtxt(network_path+'RSmask.txt')
DSmask = np.loadtxt(network_path+'DSmask.txt')
RNAname=pd.read_csv(network_path+'miRNA-name.txt')

RDORI = np.loadtxt(network_path+'RDmat.txt',delimiter=",")
RD=RDORI
DR=RD.T
DR_sparse=np.loadtxt(network_path+'DR.csv',delimiter=",")

num_R = len(RS)
num_D = len(DS)

dim_R=500
dim_D=500
dim_pred = 512
dim_pass = 500
GN=0.2  #global norm to be clipped
PNR='all'  # positive negative ratio
testS='o'  #Test scenario

diseasedict={46:'Breast neoplasms',55:'Hepatocellular carcinoma',58:'Renal cell carcinoma',60:'Squamous cell carcinoma',
             84:'Colorectal neoplasms',128:'Glioblastoma',143:'Heart failure',186:'Acute myeloid leukemia',203:'Lung neoplasms',
             220:'Melanoma',269:'Ovarian neoplasms',271:'Pancreatic neoplasms',288:'Prostatic neoplasms',320:'Stomach neoplasms',
            329:'Urinary bladder neoplasms'}
diseasedict1={288:'Prostatic neoplasms',320:'Stomach neoplasms',329:'Urinary bladder neoplasms'}
diseasedict2={60:'Squamous cell carcinoma',203:'Lung neoplasms',
             220:'Melanoma',269:'Ovarian neoplasms',271:'Pancreatic neoplasms',288:'Prostatic neoplasms',320:'Stomach neoplasms',
            329:'Urinary bladder neoplasms'}
