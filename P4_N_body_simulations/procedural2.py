# -*- coding: utf-8 -*-
# @Author: Patrice Béchard 20019173
# @Date:   2017-04-01 23:38:41
# @Last Modified time: 2017-04-09 20:51:19
#
# Procedural version of Projet 4 PHY3075
# 
#---------------------------Modules---------------------------------------------
import numpy as np 
import matplotlib.pyplot as plt
import sys
import time
from scipy.constants import pi as pi
import copy
import progressbar
from termcolor import colored

plt.style.use('patrice')
print("Execution Start Time :",time.asctime())
start=time.time()            		#time starts at beginning of execution

#-----------------------------Global Constants---------------------------------

GGRAV = 2.277e-7					#G in kpc**3/M_sun/P**2
N = 1000000							#number of star particles
RD = 4								#width of gaussian disk
MSTAR = 1.e11/N 					#masse mass of star particles
NA = 500							#number of rings
L = 30								#Domain from [[-L,L],[-L,L]]
#DR = 6*RD/NA						#width of rings
DR = L/NA
M = 64								#number of cells
DELTA = 2*L / M 					#width of cells
dt = 0.0001							#time step
nIter = 10000						#number of iterations

rdmspeed = False					#only change these lines
darkMatter = True
shape = 2							#0 : gaussian, 1: ellipse, 2: cross

if darkMatter:
	sigmaHalo = 150*MSTAR				#dark matter parameters
else:
	sigmaHalo = 0
rHalo = 15
alpha = 1.2


#--------------------------Fonctions--------------------------------------------
def init_pos():
	"""Initial configuration of stars"""
	if shape == 0:							#gaussian
		x = np.random.normal(0.,RD,N)		
		y = np.random.normal(0.,RD,N)
	elif shape == 1:						#ellipse
		x = np.random.normal(0.,RD,N)		
		y = np.random.normal(0.,RD/2,N)
	elif shape == 2:						#cross
		x1 = np.random.normal(0.,RD/2,int(N/2))		
		y1 = np.random.normal(0.,RD,int(N/2))
		x2 = np.random.normal(0.,RD,int(N/2))
		y2 = np.random.normal(0.,RD/2,int(N/2))
		x = np.append(x1,x2)
		y = np.append(y1,y2)
	return x,y

def compute_mass_distr():
	sigmaDarkMatter = sigmaHalo/(1+(np.arange(NA)*DR/rHalo)**alpha)
	deltaM = np.bincount(inring,minlength=NA)*MSTAR				#vectorized
	deltaM += 2 * pi * (np.arange(NA)*DR + 0.5*DR) * DR * sigmaDarkMatter
	Mr = np.cumsum(deltaM)										#vectorized
	return deltaM,Mr

def init_speed(radius,inring,x,y):
	temp = np.arange(NA)
	temp[0] = temp[1]
	Omega = np.sqrt(GGRAV*Mr/np.power(temp*DR,3)) #index 0 set as nan
	Omega[0] = 0
	if rdmspeed:									#keplerian or randomized speed
		vrot = Omega[inring] * radius
		vr = np.random.normal(0.,vrot*0.1,N)
		va = np.random.normal(0.,vrot*0.1,N) + 0.95*vrot
		vx = (-y*va + x*vr)/radius
		vy = ( x*va + y*vr)/radius
	else:
		vx = -Omega[inring] * y				#- va * sin (angle)
		vy =  Omega[inring] * x				#  va * cos (angle)
	return vx,vy

def compute_distance():
	"""Computing the distance for the potential computation step"""
	bar = progressbar.ProgressBar(maxval=M+1, \
    	widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()								#progress bar for user

	dist = np.zeros([M+1,M+1,M,M])			#4-d array containing distances
	for i in range(M+1):		#we compute only once instead of every iter
		bar.update(i+1)
		for j in range(M+1):				
			for k in range(M):
				for l in range(M):
					dist[i,j,k,l] = np.sqrt((i-k-0.5)**2 + (j-l-0.5)**2)

	bar.finish()
	return dist

def compute_dark_matter():
	"""Compute the dark matter grid"""
	gridDarkMatter = np.zeros([M,M])
	for i in range(M):
		for j in range(M):
			gridDarkMatter[i,j] = np.sqrt((i-M/2)**2 + (j-M/2)**2)
	gridDarkMatter *= DR 	#we have the distance from the center for each cell
	gridDarkMatter = sigmaHalo / (1 + (gridDarkMatter/rHalo)**alpha)
	return gridDarkMatter

def compute_density(x,y):
	"""Computing density of stars in each cell of the grid"""
	sigma = np.histogram2d(x,y,bins=[M,M],range=[[-L,L],[-L,L]]) #vectorized
	sigma = sigma[0] * MSTAR / (DELTA*DELTA)			#only the histogram
	return sigma

def compute_potential(sigma,d):
	"""
	Updating potential on each grid element 
	with pre-calculated distance and density
	"""
	pot = np.zeros([M+1,M+1])
	for i in range(M+1):
		for j in range(M+1):
			pot[i,j] = np.sum(sigma/d[i,j])
	pot *= -GGRAV * MSTAR * DELTA
	return pot

def compute_forces(x,y,pot):
	"""Updating forces on each particle"""
	xs = (x % DELTA) / DELTA
	ys = (y % DELTA) / DELTA
	ix = np.array((x+L)/DELTA,dtype='int')
	iy = np.array((y+L)/DELTA,dtype='int')
	fx = -((pot[ix+1,iy]-pot[ix,iy])*(1-ys) + \
						(pot[ix+1,iy+1]-pot[ix,iy+1])*ys)/DELTA
	fy = -((pot[ix,iy+1]-pot[ix,iy])*(1-xs) + \
						(pot[ix+1,iy+1]-pot[ix+1,iy])*xs)/DELTA
	return fx,fy


def verlet(x,y,vx,vy,fx,fy,fx0,fy0):
	"""Verlet algorithm for the evolution of the positions and speeds"""
	x += vx*dt + 0.5*fx0*dt*dt/MSTAR
	y += vy*dt + 0.5*fy0*dt*dt/MSTAR
	vx += 0.5 * dt * (fx0+fx)/MSTAR
	vy += 0.5 * dt * (fy0+fy)/MSTAR
	return fx,fy 								#assigned to fx0, fy0 at output

def adjust_position(x,y,vx,vy,radius,inring):
	"""Adjust position of star if out of bounds"""
	xupper = (x > L) ; xlower = (x < -L)		#bool arrays used to vectorize operations
	yupper = (y > L) ; ylower = (y < -L)
	x += xupper*(2*L - 2*x) + xlower*(-2*L - 2*x)		#change position
	y += yupper*(2*L - 2*y) + ylower*(-2*L - 2*y)
	vx += xupper*(-2*vx) + xlower*(-2*vx)				#change speed
	vy += yupper*(-2*vy) + ylower*(-2*vy)
	radius = np.sqrt(x*x + y*y)							#update radius
	inring = np.array(radius/DR,dtype='int')			#update in what ring
	return radius,inring

def show_fig417(x,y,i,xpos,ypos):
	"""Recreation of figure 4.17"""
	plt.figure(1,figsize=(9,9))
	plt.plot(x,y,'go',ms=0.25)
	for j in range(len(xpos)):
		plt.plot(xpos[j],ypos[j],lw = 2)
	temp = -15
	"""
	for i in range(self.M+1):							#grid
		plt.axhline(temp,-15,15,c='r',lw=0.5)
		plt.axvline(temp,-15,15,c='r',lw=0.5)
		temp += self.DELTA
	"""
	plt.axis([-30,30,-30,30])
	#plt.savefig("/Users/user/417.png",format='png', dpi=1000)
	plt.savefig("testfig2/417_%05d.png"%i,format='png')
	plt.close()

def show_fig421(vrot0,vrot1,i=None):

	radiusplot = np.arange(NA)*30/NA
	vrot0 = vrot0*3.086e+16/7.5e15					#convert to km/s
	vrot1 = vrot1*3.086e+16/7.5e15

	plt.figure(2,figsize=(9,9))
	plt.plot(radiusplot,vrot0,'ko',ms=0.5)
	plt.plot(radiusplot,vrot1,'ko')
	plt.ylabel(r'$v_{rot}$ (km s$^{-1}$)', color='k')
	plt.xlabel(r'$r$ (kpc)')
	plt.axis([0,30,0,max(vrot1)*1.05])

	if i != None:
		plt.savefig("testfig2/421_%d.png"%(i),format='png')
	else:
		plt.savefig("testfig2/421.png",format='png', dpi=1000)	#last one in high quality
	plt.close()

#--------------------------Main-------------------------------------------------
print("Initiating seed")
np.random.seed(1)

print("Initiating positions")
x,y = init_pos()

#initialize empty arrays
radius = np.zeros(N) ; 
inring = np.zeros(N,dtype='int')
vx = np.zeros(N) ; vy = np.zeros(N)
fx0 = np.zeros(N) ; fy0 = np.zeros(N)
radius,inring = adjust_position(x,y,vx,vy,radius,inring)	#set radius and in which ring

print("Computing mass distribution")
deltaM,Mr = compute_mass_distr()

print("Initiating speed")
vx,vy = init_speed(radius,inring,x,y)

distr = np.bincount(inring,minlength=NA)			#used to plot speed distribution
vrot0 = np.zeros(NA)
vrot = np.sqrt(vx*vx + vy*vy)
for i in range(N):
	vrot0[inring[i]] += vrot[i]
for i in range(NA):
	if distr[i] != 0:
		vrot0[i] /= distr[i]

print("Computing dark matter distribution")
gridDarkMatter = compute_dark_matter()

print(colored("Computing distance",'red'))		
dist = compute_distance()

part2x = [[] for i in range(4)]				#to track particles during simulation
part2y = [[] for i in range(4)]

for i in range(1,nIter+1):
	itertime = time.time()
	print("Iteration #",i)
	sigma = compute_density(x,y)				#compute density
	sigma += gridDarkMatter						#add dark matter contribution if any
	pot = compute_potential(sigma,dist)			#compute potential
	fx,fy = compute_forces(x,y,pot)				#compute forces
	fx0,fy0 = verlet(x,y,vx,vy,fx,fy,fx0,fy0)	#verlet algorithm
	radius,inring = adjust_position(x,y,vx,vy,radius,inring)	#update radius, in which ring
	print(time.time()-itertime)
	if i%100 == 0:
		show_fig417(x,y,i,part2x,part2y)		#position plot
		vrot = np.sqrt(vx*vx+vy*vy)
		vrot1 = np.zeros(NA)
		distr = np.bincount(inring,minlength=NA)
		for j in range(N):
			vrot1[inring[j]] += vrot[j]
		for j in range(NA):
			if distr[j] != 0:
				vrot1[j] /= distr[j]
		show_fig421(vrot0,vrot1,i)				#speed distribution plot

	for j in range(4):
		part2x[j].append(x[j])					#append current particles position
		part2y[j].append(y[j])

for i in range(len(inring)):					
	if inring[i]>=500:
		inring[i] = 499

distr = np.bincount(inring,minlength=NA)
vrot = np.sqrt(vx*vx + vy*vy)
vrot1 = np.zeros(NA)
sigma1 = np.zeros(NA)
for i in range(N):
	vrot1[inring[i]] += vrot[i]
for i in range(NA):
	if distr[i] != 0:
		vrot1[i] /= distr[i]
show_fig421(vrot0,vrot1)						#speed distribution

totaltime=time.time()-start
print("Total time : %d h %d m %d s"%(totaltime//3600,(totaltime//60)%60,totaltime%60))


