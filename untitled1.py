 # -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:15:44 2022

@author: yanbw
"""

from generateDatasets import *
from surrogatefromfile import *

import os    


ti=0
ti=000
tf=600
dt=0.01
dtS=1
TTCellModel.setSizeParameters(ti, tf, dt, dtS)

Timepoints=TTCellModel.getEvalPoints()
size=np.shape(Timepoints)[0]
Ns = 5000
hypox=cp.Uniform(0,1.25)
hyper=cp.Uniform(0,1.25)
acid=cp.Uniform(0,1.25)

dist = cp.J(hypox,hyper,acid)
samples = dist.sample(Ns)

start = timeit.default_timer()
print("--Solving")
sols=TTCellModel.run(samples.T,use_gpu=True,regen=True)

stop = timeit.default_timer()
time=stop-start
print('Time Solve: ',time) 


print("--Processing results")


start = timeit.default_timer()


wfs=np.zeros((Ns,size))

qoi={'ADP90':np.zeros(Ns),'ADP50':np.zeros(Ns),'dVmax':np.zeros(Ns),'Vreps':np.zeros(Ns),"tdV":np.zeros(Ns)}

for i in range(Ns):
   # print(sols)
    for label,v in sols[i].items():
       if(label!='Wf'):
           qoi[label][i]=v     
    for k in range(size):
        wfs[i,k]=sols[i]["Wf"][k]
    #plt.plot(Timepoints,wfs[i])
    #plt.show()


mean=np.array([np.mean(wfs.T[i]) for i in range(size)])
std=np.array([np.std(wfs.T[i]) for i in range(size)])
minw=np.array([np.min(wfs.T[i]) for i in range(size)])
maxw=np.array([np.max(wfs.T[i]) for i in range(size)])




print("--Ploting Wf UQ")

fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
ax1.plot(Timepoints,mean, label="mean")
ax1.fill_between(Timepoints,mean-std,mean+std, alpha=0.7,label="std")

n=100
for i in range(0,Ns,int(Ns/n)):
    ax2.plot(Timepoints,wfs[i])

ax1.legend(loc='best')
ax2.legend(loc='best')
plt.show()


print("--QOI")
for label,arr in qoi.items():
    print("(",label,") = ",np.mean(arr)," +- ",np.std(arr), "   ranging from ",np.min(arr),"to",np.max(arr))
    


stop = timeit.default_timer()

time=stop-start
print('Time processing results: ',time) 


