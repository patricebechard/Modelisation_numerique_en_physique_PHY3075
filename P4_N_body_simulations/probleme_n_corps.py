# -*- coding: utf-8 -*-
# @Author: Patrice Béchard
# @Date:   2017-03-20 18:07:43
# @Last Modified time: 2017-03-31 18:11:43
#
# PHY3075 - Modélisation numérique en physique
# Projet 4 - Formation de galaxies et matière sombre

#--------------------------Modules----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

Ggrav = 2.277e-7							#gravitational constant in terms of kpc^3/(m_sun * P^2)
N = 1000										#nb de particules etoiles
rD = 6										#largeur du disque gaussien, en kpc
metoile = 1.e6 / (N/1.e5)					#masse d'une particule-etoile
nIter = 1000
M = 64
dt = 0.00025
L=30
x, y, rayon, angle = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
Delta = 2 * L / M
dt = 0.05

Na = 500									#nombre d'anneaux
deltaM = np.zeros(Na)						#masse dans chaque anneau
Mr = np.zeros(Na)							#masse cumulative sous chaque anneau
Omega = np.zeros(Na)						#vitesse angulaire (keplerienne)
iann = np.zeros(N,dtype='int')				#anneau pour chaque particule
dr = 6*rD/Na 								#largeur radiale des anneaux

#----------------------------Functions------------------------------------------
def initPositions(x,y,rayon,iann,deltaM):
	for k in range(N):							#initialisation des positions
		x[k] = np.random.normal(0.,rD)
		y[k] = np.random.normal(0.,rD)
		rayon[k] = np.sqrt(x[k]*x[k] + y[k]*y[k])
		iann[k] = int(rayon[k]/dr)
		deltaM[iann[k]] += metoile

def initSpeeds(Omega,rayon,iann):
	vr = np.zeros(N)							#vitesse radiale
	va = np.zeros(N)							#vitesse azimutale
	for k in range(N):							#initialisation des vitesses
		vr[iann[k]] = 0							#orbites circulaires, v_r = 0
		va[iann[k]] = Omega[iann[k]] * rayon[iann[k]]	
	return va,vr

def cumulMass(Mr,deltaM,Omega):
	Mr[0] = deltaM[0]
	for j in range(1,Na):						#masse cumulative sous chaque anneau
		Mr[j]= Mr[j-1] + deltaM[j]
		Omega[j] = np.sqrt(Ggrav*Mr[j]/(j*dr)**3)	#eqn 4.26
	Omega[0]=Omega[1]

def computeDensity(x,y):
	sigma=np.zeros([M,M])        	# densite sur M X M cellules, initialisee a zero
	for n in range(0,N):         	# boucle sur les N particules
		k=int((x[n]+L)/Delta) 	 	# numero de cellule en x
		l=int((y[n]+L)/Delta)	  	# numero de cellule en y
		sigma[k,l] += metoile    	# cumul de la masse dans la cellule
	sigma /= Delta*Delta            # conversion en densite surfacique
	return sigma

def computePotential(sigma):	
	pot=np.zeros([M+1,M+1])				# re-initialisation du potentiel
	for k in range(0,M):				# boucles sur MXM cellules
		for l in range(0,M):
			if sigma[k,l] > 0.:
				for i in range(0,M+1):	# boucles sur (M+1)X(M+1) coins
					for j in range(0,M+1):
						d=np.sqrt((i-k-0.5)**2+(j-l-0.5)**2) # distance d(i,j,k,l)
						pot[i,j] += sigma[k,l]/d  # contribution a l’integrale
	pot *= (-Ggrav*metoile*Delta)               # les constantes a la fin
	return pot

def computeForces(pot,x):
	xs = np.zeros(N) ; ys = np.zeros(N)
	ix = [] ; iy = []
	f_x = np.zeros(N) ; f_y = np.zeros(N)
	for i in range(N):
		xs[i] = (x[i] % Delta)/Delta 			# coordonnees relatives dans cellule
		ys[i] = (y[i] % Delta)/Delta
		ix.append(int((x[i]+L)/Delta))           # coin de la cellule correspondante
		iy.append(int((x[i]+L)/Delta))
	for n in range(N):			# boucle sur les particules
		f_x[n]=-((pot[ix[n]+1,iy[n]]-pot[ix[n],iy[n]])*(1-ys[n]) + \
				 (pot[ix[n]+1,iy[n]+1]-pot[ix[n],iy[n]+1])*(ys[n]))/Delta
		f_y[n]=-((pot[ix[n],iy[n]+1]-pot[ix[n],iy[n]])*(1-xs[n]) + \
				 (pot[ix[n]+1,iy[n]+1]-pot[ix[n]+1,iy[n]])*(xs[n]))/Delta
	return f_x,f_y

def Verlet(x,y,v_a,v_r,f_x,f_y,rayon,angle,f_x0,f_y0):

	f_r0 = ( np.cos(angle)*f_x0 + np.sin(angle)*f_y0 + v_a*v_a/rayon )
	f_a0 = (-np.sin(angle)*f_x0 + np.cos(angle)*f_y0/rayon + v_r*v_a/rayon)
	# conversion cartesien a polaire
	rayon = np.sqrt(x*x + y*y)
	angle = np.arctan2(y,x)
	f_r = ( np.cos(angle)*f_x + np.sin(angle)*f_y + v_a*v_a/rayon )
	f_a = (-np.sin(angle)*f_x + np.cos(angle)*f_y/rayon + v_r*v_a/rayon)
	# algorithme de Verlet
	rayon += v_r*dt + 0.5*(f_r0/self.METOILE)* (self.dt*self.dt)
	angle += v_a*dt + 0.5*(f_a0/self.METOILE)* (self.dt*self.dt)
	v_r   += dt*(f_r0+f_r)/2.
	v_a   += dt*(f_a0+f_a)/2.
	f_r0,f_a0 = f_r,f_a
	# reconversion des positions en cartesien
	x,y = np.cos(angle)*rayon, np.sin(angle)*rayon

#-----------------------------MAIN----------------------------------------------
initPositions(x,y,rayon,iann,deltaM)
va,vr = initSpeeds(Omega,rayon,iann)
cumulMass(Mr,deltaM,Omega)

for iter in range(nIter):
	if iter%10 == 0:
		sigma = computeDensity(x,y)
		pot = computePotential(sigma)
	fx,fy = computeForces(pot,x)
	if iter == 0:
		fx0, fy0 = fx, fy
	Verlet(x,y,va,vr,fx,fy,rayon,angle,fx0,fy0)



"""
pseudocode


initialisations
vitesses initiales

for iter in range(nIter):
	# on ne va pas les calculer a chaque iter (densite et potentiel, ex 10 pas de temps)
	calculer densite Sigma(x,y)					
	calculer potentiel Phi(x,y)					#partie principale du code
	calculer forces F_x (x,y), F_y(x,y)
	transformer en coordonnees polaires (x,y) -> (r,theta)
	algorithme de Verlet
	transformer en coordonnees cartesiennes (r,theta) -> (x,y)

	(algo de verlet en polaires vs cartésien fct environ pareil on peut prendre l'un ou l'autre)


pour lundi prochain, appliquer la technique avec une grosse particule 
"""