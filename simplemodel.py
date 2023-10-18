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

l=0.99
h=1

dist = model.getDist(low=l,high=h)
samples = dist.sample(2,rule="latin_hypercube")
a=modelA.run(samples.T,use_gpu=False,regen=True,name="tS.txt")

plt.plot(tp,a[0]['Wf'],label="Model A=1")

model.setSizeParameters(ti, tf, dt, dtS)
tp=modelB.getEvalPoints()


dist = modelB.getDist(low=l,high=h)
samples = dist.sample(2,rule="latin_hypercube")
a=modelB.run(samples.T,use_gpu=False,regen=True,name="tS.txt")

plt.plot(tp,a[0]['Wf'],label="Model B=1")

modelC.setSizeParameters(ti, tf, dt, dtS)
tp=model.getEvalPoints()


dist = modelC.getDist(low=l,high=h)
samples = dist.sample(2,rule="latin_hypercube")
a=modelC.run(samples.T,use_gpu=False,regen=True,name="tS.txt")

plt.plot(tp,a[0]['Wf'],label="Model C P=1")

l=1.99
h=2
plt.legend(loc="upper right")


l=0
h=0.001


dist = modelB.getDist(low=l,high=h)
samples = dist.sample(2,rule="latin_hypercube")
a=modelB.run(samples.T,use_gpu=False,regen=True,name="tS.txt")

plt.plot(tp,a[0]['Wf'],"--")

modelC.setSizeParameters(ti, tf, dt, dtS)
tp=model.getEvalPoints()


dist = modelC.getDist(low=l,high=h)
samples = dist.sample(2,rule="latin_hypercube")
a=modelC.run(samples.T,use_gpu=False,regen=True,name="tS.txt")

plt.plot(tp,a[0]['Wf'],"--")


plt.legend(loc="upper right")

dist = model.getDist(low=l,high=h)
samples = dist.sample(2,rule="latin_hypercube")
a=modelA.run(samples.T,use_gpu=False,regen=True,name="tS.txt")

plt.plot(tp,a[0]['Wf'],"--",label="Base Model, P=0")




plt.legend(loc="upper right")


