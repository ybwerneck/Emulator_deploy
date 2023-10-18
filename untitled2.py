# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 20:33:24 2022

@author: yanbw
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:37:32 2022

@author: yanbw
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 02:58:23 2021

@author: yanbw
"""

import subprocess 
import sys
import numpy as np
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
from scipy.integrate import odeint
import lmfit
from lmfit.lineshapes import gaussian, lorentzian
import chaospy as cp
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from SALib.sample import saltelli
from SALib.analyze import sobol
import timeit
from modelTT import TTCellModel

from sklearn import linear_model as lm
import csv

#Choosing Parameters of Interest

TTCellModel.setParametersOfInterest(["gK1","gKs","gKr","gto","gNa","gCal"])

def calcula_looSingle(y, poly_exp, samples,model):
    
    
    #SERIAL LOO CALC
    nsamp = np.shape(y)[0]
    deltas = np.zeros(nsamp)
    samps = samples.T
    for i in range(nsamp):
        indices = np.linspace(0,nsamp-1,nsamp, dtype=np.int32)
        indices = np.delete(indices,i)
        subs_samples = samps[indices,:].copy()
        subs_y =[ y[i] for i in (indices)]
        subs_poly = cp.fit_regression (poly_exp,subs_samples.T,subs_y,model=model,retall=False) 
        yhat = cp.call(subs_poly, samps[i,:])
        deltas[i] = ((y[i] - yhat))**2

    y_std = np.std(y)
    err = np.mean(deltas)/np.var(y)
    acc = 1.0 - np.mean(deltas)/np.var(y)
    
f2 = open('numeric.csv', 'a',newline='')
writer2 = csv.writer(f2)
#Simulation size parameteres


ti=0
ti=5000
tf=5400
dt=0.01
dtS=1
TTCellModel.setSizeParameters(ti, tf, dt, dtS)
Timepoints=TTCellModel.getEvalPoints()
size=np.shape(Timepoints)[0]
Ns = 500

hypox=cp.Uniform(0,0.1)
hyper=cp.Uniform(0,0.1)
acid=cp.Uniform(0,1)

dist = cp.J(hypox,hyper,acid)
samples = dist.sample(Ns)

print("--Solving")

sols=TTCellModel.run(samples.T,use_gpu=True,regen=False)

wfs=np.zeros((Ns,size))

qoi={'ADP90':np.zeros(Ns),'ADP50':np.zeros(Ns),'dVmax':np.zeros(Ns),'Vreps':np.zeros(Ns)}

for i in range(Ns):
   for label,v in sols[i].items():
      if(label!='Wf'):
        qoi[label][i]=v     
   for k in range(size):
        wfs[i,k]=sols[i]["Wf"][k]






    
    #Sample the parameter distribution
    
    
    
    
    
alpha=1
eps=0.75
kws = {"fit_intercept": False,"normalize":False}
models = {
    
        
        "OLS CP": None,
        "LARS": lm.Lars(**kws,eps=eps),
     #    "OLS SKT": lm.LinearRegression(**kws),
        #"ridge"+str(alpha): lm.Ridge(alpha=alpha, **kws),
     #   "OMP"+str(alpha):
     #   lm.OrthogonalMatchingPursuit(n_nonzero_coefs=3, **kws),
        
        #"bayesian ridge": lm.BayesianRidge(**kws),
        #"elastic net "+str(alpha): lm.ElasticNet(alpha=alpha, **kws),
        #"lasso"+str(alpha): lm.Lasso(alpha=alpha, **kws),
        #"lasso lars"+str(alpha): lm.LassoLars(alpha=alpha, **kws),
        
        
    
      
    }
    
    ##
pltxs=2
pltys=0
pmin,pmax=2,4
    
while(pltys*pltxs<len(models)):
        pltys=pltys+1
        
    

for qlabel,dataset in qoi.items():
    print('\n',"QOI: ", qlabel,'\n')      
    ##Adpative algorithm chooses best fit in deegree range
        
   
          
    for label, model in models.items():   
            print('\n--------------',"\n")
            print("Beggining ", label)
            loos= np.zeros((pmax-pmin+1))
            gF= np.zeros((pmax-pmin+1))
            timeL= np.zeros((pmax-pmin+1))
            
            startT=timeit.default_timer()
            pols=[]
            
            print(np.shape(samples.T))
            for P in list(range(pmin,pmax+1,1)):             
             
                print('\n')
                print('D=',P)
                
                #generate and fit expansion            
              
                
              
                start = timeit.default_timer()
                
                
                poly_exp = cp.generate_expansion(P, dist,rule="three_terms_recurrence")
                fp = cp.fit_regression (poly_exp,samples,dataset,model=model)  
                
                stop = timeit.default_timer()
                time=stop-start
                print('Time to generate exp: ',time) 
                gF[P-pmin]=time
             
                #calculate loo error
                
                start = timeit.default_timer()
                
                loos[P-pmin]=calcula_looSingle(dataset,poly_exp,samples,model)
                
                stop = timeit.default_timer()
                timeL[P-pmin]=stop-start
                print('Time to LOO: ',timeL[P-pmin],'LOO: ',loos[P-pmin]) 
    
    
                pols.append(fp)
                
                print('\n')
            
            stopT = timeit.default_timer()
            TT=stopT-startT
            #Choose best fitted poly exp in degree range->lowest loo error
            degreeIdx=loos.argmin()
            loo=loos[degreeIdx]
            fitted_polynomial=pols[degreeIdx]
            ##
            print('AA picked D= ',degreeIdx+pmin," Generate Validation Results") 
            
            ##Calculate Sobol Error
            #s1f=np.array(sensitivity['S9'])
            
            #Caluclate Validation Error
            
                        
            row2=['Chaos Py',qlabel,label,Ns,'-',degreeIdx+pmin,gF[degreeIdx]]
   
            writer2.writerow(row2)
            
            
            print('--------------',"\n")
            
 










print("--Ploting Wf")

for waveform in wfs:
        
    plt.plot(Timepoints,waveform, label="mean")
plt.show()


print("--QOI")
for label,arr in qoi.items():
    print("(",label,") = ",np.mean(arr)," +- ",np.std(arr), "   ranging from ",np.mean(arr),"to",np.max(arr))
    
