import os                                                              
import time   
import pickle
import sys
import numpy as np                                                     
import pandas as pd   
import seaborn as sns                                                  
import matplotlib.pyplot as plt  
import itertools                                          
                                                                       
from sklearn.metrics import mean_absolute_error                        
from sklearn.metrics import mean_squared_error                         
from sklearn.model_selection import train_test_split                   
from sklearn.preprocessing import StandardScaler                       
from sklearn.model_selection import train_test_split         

import ase                                                             
import ase.build        
import math                                               
from ase import Atoms                                                  
from ase.atoms import Atoms                                            
from ase.io import read, write                                         
from ase.calculators.dftb import Dftb                                  
from ase.units import Hartree, mol, kcal, Bohr                         
                                                                       
from Calculator import src_nogrd                                                       
from Calculator.src_nogrd import sym_func_show                                    
from Calculator.src_nogrd import xyzArr_generator                                 
from Calculator.src_nogrd import feat_scaling_func                                
from Calculator.src_nogrd import at_idx_map_generator
from Calculator.src_nogrd import at_idx_map_generator_old
                                                                                                                           
import pickle
from itertools import combinations_with_replacement as comb_replace
                                                                       
import Utils.DirNav                                                
from Utils.dftb_traj_io import read_scan_traj

import pickle

from keras import models
from keras import layers
from keras import optimizers
from keras.optimizers import Adam


np.random.seed(98167)  # for reproducibility       
geom_filename          = os.path.join('/home/lgomez/dftb-nn/Charges/TDstructures.xyz')
md_train_arr_origin    = read_scan_traj(filename=geom_filename)
md_train_arr = md_train_arr_origin.copy(deep=False).reset_index(drop=True)

nAtoms, xyzArr = xyzArr_generator(md_train_arr)# Calculate distance dataframe from xyz coordinates
distances = src_nogrd.distances_from_xyz(xyzArr, nAtoms)

SUPPORTED_ELEMENTS = ['H', 'C', 'S']
file_count=13548
at_idx_map = at_idx_map_generator_old(md_train_arr[0])

Cr=1.41
Hr=0.59
Sr=1.48

p=np.sqrt(math.pi)

Cj=0.1824
Hj=0.2098
Sj=0.1644

CCd=(1/Cr*p)+Cj
HHd=(1/Hr*p)+Hj
SSd=(1/Sr*p)+Sj

CH=np.sqrt(np.square(Cr)+np.square(Hr))
CS=np.sqrt(np.square(Cr)+np.square(Sr))
SH=np.sqrt(np.square(Sr)+np.square(Hr))
CC=np.sqrt(np.square(Cr)+np.square(Cr))
SS=np.sqrt(np.square(Sr)+np.square(Sr))
HH=np.sqrt(np.square(Hr)+np.square(Hr))

CHs=list(tuple(sorted(pair)) for pair in itertools.product(at_idx_map['H'], at_idx_map['C']))
SHs=list(tuple(sorted(pair)) for pair in itertools.product(at_idx_map['H'], at_idx_map['S']))
SCs=list(tuple(sorted(pair)) for pair in itertools.product(at_idx_map['C'], at_idx_map['S']))
CCs=list(tuple(sorted(pair)) for pair in itertools.product(at_idx_map['C'], at_idx_map['C']))
HHs=list(tuple(sorted(pair)) for pair in itertools.product(at_idx_map['H'], at_idx_map['H']))
SSs=list(tuple(sorted(pair)) for pair in itertools.product(at_idx_map['S'], at_idx_map['S']))
del CCs[0]
del SSs[0]
del SSs[3]
del CCs[3]
del HHs[0]
del HHs[9]
del HHs[18]
del HHs[27]
del HHs[36]
del HHs[45]
del HHs[54]
del HHs[63]
del CCs[-1]
del SSs[-1]
del HHs[-1]

Matrix = np.zeros((file_count,15,15))
for i in range(file_count):
  Z = np.zeros((15, 15))
  for x in range (len(SCs)):
      Z[SCs[x]]=(math.erf(distances[(SCs[x])][i]/CS))/distances[(SCs[x])][i]
      Z[SCs[x][::-1]]=(math.erf(distances[(SCs[x])][i]/CS))/distances[(SCs[x])][i]
  for x in range (len(SHs)):
      Z[SHs[x]]=(math.erf(distances[(SHs[x])][i]/SH))/distances[(SHs[x])][i]
      Z[SHs[x][::-1]]=(math.erf(distances[(SHs[x])][i]/SH))/distances[(SHs[x])][i]
  for x in range (len(CHs)):
      Z[(CHs[x])]=(math.erf(distances[(CHs[x])][i]/CH))/distances[(CHs[x])][1]
      Z[(CHs[x][::-1])]=(math.erf(distances[(CHs[x])][i]/CH))/distances[(CHs[x])][i]
  for x in range (len(CCs)):
      Z[(SSs[x])]=(math.erf(distances[(SSs[x])][i]/SS))/distances[(SSs[x])][i]
      Z[(SSs[x][::-1])]=(math.erf(distances[(SSs[x])][i]/SS))/distances[(SSs[x])][i]
      Z[(CCs[x])]=(math.erf(distances[(CCs[x])][i]/CC))/distances[(CCs[x])][i]
      Z[(CCs[x][::-1])]=(math.erf(distances[(CCs[x])][i]/CC))/distances[(CCs[x])][i]
  for x in range (len(HHs)):
      Z[(HHs[x])]=(math.erf(distances[(HHs[x])][i]/HH))/distances[(HHs[x])][i]
      Z[(HHs[x][::-1])]=(math.erf(distances[(HHs[x])][i]/HH))/distances[(HHs[x])][i]
  S=[CCd,SSd,HHd,HHd,HHd,CCd,SSd,HHd,HHd,HHd,CCd,SSd,HHd,HHd,HHd]
  np.fill_diagonal(Z,S)
  Matrix[i]=Z

inverse = np.linalg.inv(Matrix)

col_Names= ['structure']
s=[str(x) for x in range(144)]
s+= ['electro']
s+=['qq']
s+=['index']
col_Names+=s

big= pd.read_csv('/home/lgomez/dftb-nn/Charges/Temp/big.csv',names=col_Names) 

ap1=big[(big['electro'] > 0.5)]
ap2=big[(big['electro'] < 0)]

dd=big.append([ap1]*10, ignore_index=True)
d=dd.append([ap2]*2, ignore_index=True)

np.random.seed(98167)  # for reproducibility       

from random import shuffle
import random
def shuffle(st,p):
    v=int((st*(1-p))/2)
    t=int(st*p)
    r=np.arange(st)
    train=random.sample(set(r), t)
    remain=np.delete(r, train)
    test=random.sample(set(remain), v)
    s=np.concatenate((train,test), axis=None)
    u = np.delete(r, s)
    val=list(u)
    return train,test,val

split=shuffle(len(d),0.8)

train = d[d["structure"].isin(["%.0f" % i for i in split[0]])]
x_train = train.drop(columns=['structure','qq','electro','index'])
y_train = np.asarray(train['electro']) 

test = d[d["structure"].isin(["%.0f" % i for i in split[1]])]
x_test = test.drop(columns=['structure','qq','electro','index'])
y_test = np.asarray(test['electro']) 
index_test = test[["structure","index","qq"]].copy()

val = d[d["structure"].isin(["%.0f" % i for i in split[2]])]
x_val = val.drop(columns=['structure','qq','electro','index'])
y_val = np.asarray(val['electro']) 
index_val = val[["structure","index","qq"]].copy()

def network(capa,neurons_hidden,neurons_initial):
    network=models.Sequential()
    network.add(layers.Dense(neurons_initial, activation='tanh', input_shape=(144,)))
    for i in range(capa):
        network.add(layers.Dense(neurons_hidden, activation='tanh'))
    network.add(layers.Dense(1))
    return network

net=network(2,15,144)

net.compile(optimizer=Adam(lr=0.0001,decay=0.001),
                            loss='mean_squared_error',
                            metrics=['mean_squared_error'])
history = net.fit(x_train, y_train,
                                validation_data=(x_val, y_val),
                                epochs=180,shuffle=True,verbose=0,                                                                  
                                batch_size=64)

net.compile(optimizer=Adam(lr=0.001,decay=0.01),
                            loss='mean_squared_error',
                            metrics=['mean_squared_error'])

history2 = net.fit(x_train, y_train,
                                validation_data=(x_val, y_val),
                                epochs=100,shuffle=True,verbose=0,
                                batch_size=32)
net.compile(optimizer=Adam(lr=0.0001,decay=0.001),
                            loss='mean_squared_error',
                            metrics=['mean_squared_error'])
history3 = net.fit(x_train, y_train,
                                validation_data=(x_val, y_val),
                                epochs=80,shuffle=True,verbose=0,                                                                  
                                batch_size=64)

net.compile(optimizer=Adam(lr=0.001,decay=0.01),
                            loss='mean_squared_error',
                            metrics=['mean_squared_error'])

history4 = net.fit(x_train, y_train,
                                validation_data=(x_val, y_val),
                                epochs=50,shuffle=True,verbose=0,
                                batch_size=32)

y_pred=net.predict(x_test)
index_test['pred'] = np.array(y_pred)  
estru=index_test['structure'].tolist()
mylist = list(dict.fromkeys(estru))
for i in mylist:
    globals()["L"+str(i)] = []
    globals()["q"+str(i)] = []
    globals()["qq"+str(i)] = []
    for s in range (0,15):
        is_2002 = index_test[(index_test['structure']==i) & (index_test['index']==s)]
        globals()["L"+str(i)].append(is_2002["pred"].mean())
        globals()["qq"+str(i)].append(is_2002["qq"].mean())
        globals()["q"+str(i)]=np.matmul(inverse[i][s], globals()["L"+str(i)])
        loss=mean_squared_error(globals()["qq"+str(i)][s], globals()["q"+str(i)][s]) 


prediction=np.reshape(y_pred,y_test.shape)

print(loss)

df=pd.DataFrame(columns=['cargas_real', 'cargas_pred'])
for s in mylist:
    real = np.concatenate([globals()["qq"+str(i)]])
    pred = np.concatenate([globals()["q"+str(i)]])
df['cargas_real']=real
df['cargas_pred']=pred

x2 = df['cargas_real']
x1 = df['cargas_pred']

x = np.random.uniform(min(df['cargas_real']),max(df['cargas_real']),size=100)

plt.figure(figsize=(7,5))
plt.plot(x, x, dashes=[10, 5, 20, 5],color='navy',linewidth=1)

ax=sns.kdeplot(x2,x1, cmap='plasma_r',shade=True, shade_lowest=False)
plt.title('Test Set Prediction H', fontsize=20)
plt.ylabel('Reference charges', fontsize = 16)
plt.xlabel('Prediction charges',fontsize = 16)
plt.tick_params(axis='both', labelsize=14)
plt.savefig("TESTSET",dpi=300)



plt.figure(figsize=(7,5))
y_predT=net.predict(x_train)
predy=np.reshape(y_predT,y_train.shape)
x3=pd.Series(predy,name="PredictedTrain")
x4=pd.Series(y_train,name="ObservedTrain")

data2 = pd.concat([x3, x4], axis=1)
g = sns.lmplot("ObservedTrain", "PredictedTrain", data2,
            scatter_kws={"marker": ".", "color": "turquoise", "alpha": 0.4 },
            line_kws={"linewidth": 1, "color": "orange"},
            height=8, aspect=1);
plt.plot(ls="--", c=".1")
plt.grid(which='major', linestyle='-', linewidth='0.5', color='purple')
plt.grid(which='minor', linestyle='-', linewidth='0.5', color='blue', alpha=0.25)
plt.minorticks_on()
plt.title('Training Set Prediction', fontsize=20)
plt.savefig("TrainSET",dpi=300)


