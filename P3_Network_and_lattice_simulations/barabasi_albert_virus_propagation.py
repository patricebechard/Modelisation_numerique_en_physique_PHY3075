# -*- coding: utf-8 -*-
"""
PHY3075 - Modele de Barabasi-Albert
Patrice Bechard
mars 2017
"""
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
import copy
from matplotlib import animation
import numpy as np

start=time.time()

N0=2                #size of network at beginning
N=1000              #final size of network
nIter=100
probConnect=0.5
probInfect=0.5
"""
epidemic models (etype):
    1 : SIR
    2 : SEIR
    3 : SIRS
"""  
etype=3       

[S,E,I,R]=[[0 for i in range(N)] for j in range(4)]
thresE=5 ; thresI=10 ; thresR=5

#--------------------------FONCTIONS-------------------------     
def create_network(G):
    for i in range(N0):
        G.add_node(i,status='S')
        S[i]=1
    G.add_edge(0,1)                         #we create the starting nodes
    for i in range(N0,N):
        connection=random.randrange(G.order())
        G.add_node(i,status='S')
        S[i]=1
        if random.random()<probConnect:
            G.add_edge(i,connection)
        else:
            G.add_edge(i,random.choice(G.neighbors(connection)))

def evolution(G):
    if etype==1:
        SIR(G)
    elif etype==2:
        SEIR(G)
    elif etype==3:
        SIRS(G)
    else:
        raise Exception ("Invalid epidemic model")
            
def SIR(G):
    temp_I=copy.copy(I)
    infect_neighbors(G)    
    update_list(I,temp_I,thresI,R,'R')          
    return
 
def SEIR(G):
    temp_E=copy.copy(E)
    temp_I=copy.copy(I)
    infect_neighbors(G)   
    update_list(E,temp_E,thresE,I,'I')
    update_list(I,temp_I,thresI,R,'R')          
    return

def SIRS(G):
    temp_I=copy.copy(I)
    temp_R=copy.copy(R)
    infect_neighbors(G)    
    update_list(I,temp_I,thresI,R,'R')
    update_list(R,temp_R,thresR,S,'S')           
    return
    
def infect(G,node,param=None):
    if etype!=2 or param is not None:
        G.node[node]['status']='I'
        I[node]=1
        S[node]=0
    else:
        G.node[node]['status']='E'
        E[node]=1
        S[node]=0

def update_list(status,copie,thres,nextstatus,change):
    for j in range(N):
        if status[j]>0 and status[j]==copie[j]:
            status[j]+=1
            if status[j]==thres:
                status[j]=0
                nextstatus[j]=1
                G.node[j]['status']=change

def infect_neighbors(G):
    Gprime=copy.deepcopy(G)
    for i in range(N):
        if Gprime.node[i]['status']=='I':
            for j in range(len(G.neighbors(i))):
                if Gprime.node[G.neighbors(i)[j]]['status']=='S'\
                        and random.random()<probInfect:
                            infect(G,G.neighbors(i)[j])
"""
dom=[i for i in range(2,11)]
dom2=copy.deepcopy(dom)
yes=[]
for thresE in dom:
    for thresI in dom2:
        value=0
        for nice in range(10):
            G=nx.Graph()
            create_network(G)
            pos=nx.spring_layout(G)    
            infect(G,random.randrange(N),1)
            values=np.zeros(N)
            #nx.draw_networkx(G,pos=pos,cmap='cool',node_color=values,node_size=10,with_labels=False)
            progress=np.array([])
            listS,listE,listI,listR=[],[],[],[]
            for i in range(N):
                if S[i]>0:
                    listS.append(i)
                elif E[i]>0:
                    listE.append(i)
                elif I[i]>0:
                    listI.append(i)
                elif R[i]>0:
                    listR.append(i)
            dataS,dataE,dataI,dataR=[len(listS)],[len(listE)],[len(listI)],[len(listR)]
            for i in range(nIter):
                evolution(G)
                listS,listE,listI,listR=[],[],[],[]
                for i in range(N):
                    if S[i]>0:
                        listS.append(i)
                    elif E[i]>0:
                        listE.append(i)
                    elif I[i]>0:
                        listI.append(i)
                    elif R[i]>0:
                        listR.append(i)
                dataS+=[len(listS)]
                dataE+=[len(listE)]
                dataI+=[len(listI)]
                dataR+=[len(listR)]
            
            value+=dataS[-1]/N

            plt.plot(np.linspace(0,nIter,nIter+1),dataS,'g',label='S')
            plt.plot(np.linspace(0,nIter,nIter+1),dataE,'b--',label='E')
            plt.plot(np.linspace(0,nIter,nIter+1),dataI,'r:',label='I')
            plt.plot(np.linspace(0,nIter,nIter+1),dataR,'k-.',label='R')
            plt.xlabel('Itération')
            plt.ylabel('Nombre')
            plt.text(0,1.08*N,r'$p_C = $%.2f , $p_I = $%.2f, $t_E = $%d , $t_I = $%d'%(probConnect,probInfect,thresE,thresI))
            plt.axis([0,nIter,0,N*1.05])
            plt.legend(fancybox=True,shadow=True)
            value+=dataS[-1]/N
                
            if etype==1:
                plt.savefig('distSIR%d_%d_%d.png'%(N,nIter,int(probInfect*100)))
            elif etype==2:
                plt.savefig('distSEIR%d_%d_%d.png'%(N,nIter,int(probInfect*100)))
            elif etype==3:
                plt.savefig('distSIRS%d_%d_%d.png'%(N,nIter,int(probInfect*100)))
            plt.show()
            
        value=value/10
        print(value)
"""
G=nx.Graph()
create_network(G)
pos=nx.spring_layout(G)    
infect(G,random.randrange(N),1)

def show_network(x):
    print('hey')
    evolution(G)
    listS,listE,listI,listR=[],[],[],[]
    for i in range(N):
        if S[i]>0:
            listS.append(i)
        elif E[i]>0:
            listE.append(i)
        elif I[i]>0:
            listI.append(i)
        elif R[i]>0:
            listR.append(i)
    print(x,"  elapsed : ",time.time()-start)
    nx.draw_networkx(G,pos=pos,node_color='g',node_size=10,with_labels=False,nodelist=listS,width=0.3)
    nx.draw_networkx(G,pos=pos,node_color='y',node_size=10,with_labels=False,nodelist=listE,width=0.3)
    nx.draw_networkx(G,pos=pos,node_color='r',node_size=10,with_labels=False,nodelist=listI,width=0.3)
    nx.draw_networkx(G,pos=pos,node_color='b',node_size=10,with_labels=False,nodelist=listR,width=0.3)

fig = plt.gcf()
anim = animation.FuncAnimation(fig, show_network, frames=nIter, interval=200)
#fig.legend(fancybox=True,shadow=True,handles=['g.','y.','r.','b.'],labels=['S','E','I','R'])

anim.save('barabasi%d_%d.mp4'%(N,nIter))  

plt.show()

