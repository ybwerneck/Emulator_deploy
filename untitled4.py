# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 23:56:48 2022

@author: yanbw
"""
import numpy as np
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import odeint
import lmfit
from lmfit.lineshapes import gaussian, lorentzian
import chaospy as cp
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from SALib.sample import saltelli
from SALib.analyze import sobol
import timeit
import os
import utils
sys.path.append('repo/CUDA-OPENMP-TTCELL-MODEL/')
from modelTT  import TTCellModel as modelA
from ModelB import TTCellModelExt as modelB
from ModelC import TTCellModelChannel as modelC

from strategycomparison import surrogatefromSet as strategycomparisonS
from surrogatefromfile import surrogatefromfile as PCEsurrogate
from Models import GPModel,PCEModel,NModel
from generateDatasets import *
from functools import partial
from sklearn import linear_model as lm
from utils import *

from copy import copy

def getPlots(a,b):
    fig,plotslot=plt.subplots(a,b,figsize=(10, 8),dpi=250)
    plotsaux=[]
    plots=[]
            
    try:
                   for row in plotslot:
                       for frame in row:
                           plotsaux.append(frame)
    except:
                   try :
                       for frame in plotslot:
                           plotsaux.append(frame)
                   except:
                           plotsaux.append+++(plotslot)
    
               
    for i in range(0,len(plotsaux)):
                plots.append(plotsaux.pop())
    
    return fig,plots
    

    
qois=["ADP90","ADP50","dVmax","Vrest"]
models={
           
           
           
           "Model A":["./compA/", modelA],
           "Model B":["./compB/", modelB],
           "Model C":["./compC/", modelC],
           
           
           
   }
           

for modelName,mdata in models.items():
    print("Beginning ", modelName," Sobol")
    mdata[1].setSizeParameters(20000,20500, 0.01, 1)
    nPar=mdata[1].getNPar()
  
    sobol1={}
    sobol2={}
    problem = {
    'num_vars': nPar,
   
    'bounds': [[0,1] for i in range(0,nPar) ]
    }
    
    N=4000
    samples = saltelli.sample(problem, N,calc_second_order=False).T
    
    Ns=np.shape(samples)[1]

    
    start = timeit.default_timer()
    sols= mdata[1].run(samples.T,use_gpu=True,regen=True,name="tS.txt")
    stop = timeit.default_timer()
    
    
    print("time model ",modelName," ",stop-start,"s")
    
    yv={}
    yv["ADP50"]=[sols[i]["ADP50"] for i in range(Ns)]
    yv["ADP90"]=[sols[i]["ADP90"] for i in range(Ns)]
    yv["dVmax"]=[sols[i]["dVmax"] for i in range(Ns)]
    yv["Vrest"]= [sols[i]["Vreps"] for i in range(Ns)]



    folder,model=mdata

    

    
    for qoi in qois:
        y=np.array(yv[qoi])
        sr=sobol.analyze(problem,y,calc_second_order=False)
        sobol1[qoi]=sr['S1']
        sobol2[qoi]=sr['ST']




    fig,plots=getPlots(2,2)
    fig.suptitle(modelName +" Sobol ")
    for qoi in qois:
        
        plot=plots.pop()
        a=[]
       
        plot.bar([str(i) for i in range(0,nPar)],sobol1[qoi])        
 
        plot.set_title(qoi)           
        plot.set(ylabel="Sobol First Order ")
    
 
    plt.savefig("sobol/"+modelName+"s1.png")
    
    for qoi in qois:
        y=np.array(yv[qoi])
        sr=sobol.analyze(problem,y,calc_second_order=False)
        sobol1[qoi]=sr['S1']
        sobol2[qoi]=sr['ST']




    fig,plots=getPlots(2,2)
    fig.suptitle("sobol/"+modelName +" Sobol ")
    for qoi in qois:
        
        plot=plots.pop()
        a=[]
       
        plot.bar([str(i) for i in range(0,nPar)],sobol2[qoi])        
 
        plot.set_title(qoi)           
        plot.set(ylabel="Sobol Total Order ")
    
    plt.savefig("sobol/"+modelName+"st.png")
    