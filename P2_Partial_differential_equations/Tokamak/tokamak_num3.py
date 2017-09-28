# -*- coding: utf-8 -*-
"""
tokamak, num 3
"""

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

#files to read

#files,analysis=['num3data_10.txt','num3data_save.txt','num3data_20.txt'],'H_0'
#files,analysis=['num3data_chi4.txt','num3data_save.txt','num3data_chi12.txt','num3data_chi16.txt'],'\chi_1'
#files,analysis=['num3data_sig2.txt','num3data_save.txt','num3data_sig05.txt'],'\sigma'
files,analysis=['num3data_step05.txt','num3data_step1.txt','num3data_step2.txt'],'step, L'

plt.figure(figsize=(9,6))
plt.loglog()
color = ['r','g','b','m']       #we can iterate over plot properties
lines=['-','--','-.',':']

#values=[10,14.4,20]            #for legend
#values=[4,8,12,16]
#values=[2.0,1.0,0.5]
values=[1.0,2.0,4.0]

linecycler=cycle(color)
linecycler2=cycle(values)
linecycler3=cycle(lines)

for file in files:

    f=open(file)

    #read the messy data files
    timenum3=[float(x) for x in (f.readline()[:-2]).split(' ') if x.replace('.','').isdigit()]
    sigtime=[float(x) for x in (f.readline()[:-2]).split(' ') if x.replace('.','').isdigit()]
    posnum3=[float(x) for x in (f.readline()[:-2]).split(' ') if x.replace('.','').isdigit()]
    sigpos=[float(x) for x in (f.readline()[:-2]).split(' ') if x.replace('.','').isdigit()]
    
    """
    vitesse=[]
    for i in range(len(timenum3)-1):
        vit=np.abs((posnum3[i+1]-posnum3[i])/(timenum3[i+1]-timenum3[i]))
        vitesse.append(vit)
    
    plt.plot(timenum3[:-1],vitesse,'.')
    timenum3=timenum3[:-1]
    """
    c=next(linecycler)
    ligne=next(linecycler3)

    plt.errorbar(timenum3, posnum3, xerr=sigtime, yerr=sigpos, ls='',color=c)

    x,y=np.log10(timenum3),np.log10(posnum3)
    
    coefficients = np.polyfit(x, y, 1)          #fit for the data
    polynomial = np.poly1d(coefficients)
    varx=[1e-1,6e1]
    ys = 10**(polynomial(np.log10(varx)))
    coeff=10**coefficients[1]
    power=coefficients[0]

    plt.plot(varx, ys,ls=ligne,color=c,label='$%s $=%.1f : x(t)=$%.3ft^{%.3f}$'%(analysis,next(linecycler2),coeff,power))


plt.xlabel(r'$t$')
plt.ylabel(r'$x$')
plt.axis([2e-1,6e1,2e-1,1.5e0])
plt.legend(fancybox=True,shadow=True)

plt.savefig('figures/num3_step.png')
plt.show()