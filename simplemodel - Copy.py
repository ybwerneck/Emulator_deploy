# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 22:44:43 2021

@author: yanbw
"""

from ModelB import TTCellModelExt as model

import timeit
import matplotlib.pyplot as plt
import warnings
import chaospy as cp

import sys

from modelTT  import TTCellModel as modelA
from ModelB import TTCellModelExt as modelB
from ModelC import TTCellModelChannel as modelC

##Simple model use, usefull for
ti=0
ti=000
tf=500
dt=0.01
dtS=1


modelA.setSizeParameters(ti, tf, dt, dtS)
tp=model.getEvalPoints()


dist = model.getDist(low=0,high=0.1)
samples = dist.sample(2,rule="latin_hypercube")
a=modelA.run(samples.T,use_gpu=False,regen=True,name="tS.txt")

plt.plot(tp,a[0]['Wf'],label="AP Waveform")
plt.axhline(a[0]["Vreps"],ti,tf,linestyle='dashed',color="green",label="V Repos")
plt.axvline(a[0]["ADP90"],linestyle='dashed',color="red",label="ADP90")
plt.axvline(a[0]["ADP50"],linestyle='dashed',color="pink",label="ADP50")
plt.axvline(a[0]["tdV"],linestyle='dashed',color="grey",label="max dV")


model.setSizeParameters(ti, tf, dt, dtS)
tp=modelB.getEvalPoints()


dist = model.getDist(low=0,high=0.1)
samples = dist.sample(2,rule="latin_hypercube")
a=modelB.run(samples.T,use_gpu=False,regen=True,name="tS.txt")

plt.plot(tp,a[0]['Wf'],label="AP Waveform")
plt.axhline(a[0]["Vreps"],ti,tf,linestyle='dashed',color="green",label="V Repos")
plt.axvline(a[0]["ADP90"],linestyle='dashed',color="red",label="ADP90")
plt.axvline(a[0]["ADP50"],linestyle='dashed',color="pink",label="ADP50")
plt.axvline(a[0]["tdV"],linestyle='dashed',color="grey",label="max dV")

modelC.setSizeParameters(ti, tf, dt, dtS)
tp=model.getEvalPoints()


dist = modelC.getDist(low=0,high=0.1)
samples = dist.sample(2,rule="latin_hypercube")
a=model.run(samples.T,use_gpu=False,regen=True,name="tS.txt")

plt.plot(tp,a[0]['Wf'],label="AP Waveform")
plt.axhline(a[0]["Vreps"],ti,tf,linestyle='dashed',color="green",label="V Repos")
plt.axvline(a[0]["ADP90"],linestyle='dashed',color="red",label="ADP90")
plt.axvline(a[0]["ADP50"],linestyle='dashed',color="pink",label="ADP50")
plt.axvline(a[0]["tdV"],linestyle='dashed',color="grey",label="max dV")


plt.legend(loc="upper right")




