# -*- coding: utf-8 -*-
"""
tokamak num 4
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from itertools import cycle

def fig2_15(init):
    """Figure 2.15 from notes"""
    dtdx=np.linspace(-6,0,601)
    lines = ['k','k--','k-.']
    linecycler=cycle(lines)
    
    for chi1 in [8,2,0.5]:
        chi=init[2]+chi1/(1+init[0]*dtdx**init[1])
        Q=-chi*dtdx
        plt.plot(-dtdx,Q,next(linecycler),label=r'$\chi_1 = $%.1f'%chi1)
    
    plt.plot(-dtdx,10*(-dtdx),'r--')
    plt.plot(-dtdx,1*(-dtdx),'r--')
    plt.xlabel(r'$-dT/dx$')
    plt.ylabel(r'$Q$')
    plt.axis([0,6,0,6])
    plt.legend(shadow=True,fancybox=True)
    plt.text(0.65, 5.6, 'Mode L',color='r')
    plt.text(4, 3.6, 'Mode H',color='r')
    plt.text(1.5, .25, r'$\alpha = 1, \beta = 4, \chi_0 = 1$')
    plt.savefig('yaya.png')
    plt.show()
    return
    
#-----------------------------MAIN-------------------------

files=['datanum4_2_chi2.txt','datanum4_2_chi8.txt','datanum4_20_chi2.txt','datanum4_20_chi8.txt']
count=0
plt.figure(figsize=(9,6))

#we plot data for each data file

for file in files:
    count+=1
    f=open(file)
    data=(f.read())
    time=str(data).replace('\n','').split('][')
    for i in range(len(time)):
        time[i]=time[i].replace(']','')
    
    time1=time[0].split(' ')
    values1=time[1].split(' ')
    
    time2=[]
    values2=[]
    for i in time1:
        try:
            time2.append(float(i))
        except:
            pass
    for i in values1:
        try:
            values2.append(float(i))
        except:
            pass
    
    print(len(time2))
    print(len(values2))
    if count==1:
        plt.plot(time2,values2,'m+',label=r'$H_0=2,\chi_1=2$')
    elif count==2:
        plt.plot(time2,values2,'gx',label=r'$H_0=2,\chi_1=8$')
    elif count==3:
        plt.plot(time2,values2,'bv',label=r'$H_0=20,\chi_1=2$')
    else:
        plt.plot(time2,values2,'y^',label=r'$H_0=20,\chi_1=8$')

alpha=1
beta=4
chi0=1
chi1=8
init=[alpha,beta,chi0,chi1]     #to compute chi

fig2_15(init)
