# -*- coding: utf-8 -*-
# @Author: Patrice Béchard 20019173
# @Date:   2017-03-27 19:06:53
# @Last Modified time: 2017-04-01 11:15:01
#
# PHY3075 - Modélisation numérique en physique
# Projet 4 - La matière sombre

#-------------------------------Modules-----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy.constants import pi

plt.style.use('patrice')

nIter = 1000

start = time.time()

#-------------------------------Classes-----------------------------------------
class Galaxy:
	"""Contains all stars of the system"""

	GGRAV = 2.277e-7						#in kpc^3/(m_sun * P^2)
	N = 100000								#Number of stars in galaxy
	RD = 6									#width of gaussian disk
	NA = 500								#nombre d'anneaux
	MSTAR = 1.e6 / (N/1.e5)					#mass of a star
	DR = 6*RD/NA 							#largeur radiale des anneaux
	M = 64									#number of squares by line
	L = 30									#Domain from [[-L,L],[-L,L]]
	dt = 0.00025
	DELTA = 2*L / M

	class Star:
		"""Contains every information about a particular star"""
		def __init__(self):
			"""Constructor of the Star class"""
			self.x = np.random.normal(0., Galaxy.RD)
			self.y = np.random.normal(0., Galaxy.RD)
			self.adjust_position()
			self.vx = 0
			self.vy = 0
			self.fx = 0
			self.fy = 0
			self.fx0 = 0
			self.fy0 = 0
			self.area_density = 0

		def adjust_position(self):
			"""Adjust position of star if out of bounds"""
			if self.x > Galaxy.L:					
				self.x = 2*Galaxy.L - self.x
			elif self.x < -Galaxy.L:
				self.x = -2*Galaxy.L - self.x
			if self.y > Galaxy.L:					
				self.y = 2*Galaxy.L - self.y
			elif self.y < -Galaxy.L:
				self.y = -2*Galaxy.L - self.y
			self.radius = np.sqrt(self.x*self.x + self.y*self.y)
			self.angle = np.arctan(self.y/self.x)
			self.inring = int(self.radius/Galaxy.DR)

		def update_position(self,pos):
			return

		def update_velocity(self,v):
			return

	def __init__(self):
		"""Constructor of the Galaxy class"""
		self.galaxy = []
		self.omega = np.zeros(self.NA)
		self.Mr = np.zeros(self.NA)
		self.deltaM = np.zeros(self.NA)
		self.sigma = np.zeros([self.M,self.M])
		self.pot = np.zeros([self.M+1,self.M+1])
		for i in range(self.N):
			self.galaxy.append(self.Star())
			self.deltaM[self.galaxy[-1].inring] += self.MSTAR	

	def __getitem__(self,i):
		return self.galaxy[i]

	def compute_mass(self):
		self.Mr[0] = self.deltaM[0]
		for i in range(1,self.NA):
			self.Mr[i] = self.Mr[i-1] + self.deltaM[i]
			self.omega[i] = np.sqrt(self.GGRAV * self.Mr[i] / (i*self.DR)**3)
		self.omega[0] = self.omega[1]
		self.compute_area_density()

	def compute_area_density(self):
		"""Computes area density in terms of distance from center"""
		for star in self.galaxy:
			star.area_density = self.deltaM[star.inring] / \
									(2*pi*star.radius*self.DR)

	def init_velocity(self):
		for star in self.galaxy:
			va = self.omega[star.inring] * star.radius
			star.vx = -va * np.sin(star.angle)
			star.vy =  va * np.cos(star.angle)
			#star.update_velocity([vr,va])

	def compute_density(self):
		for star in self.galaxy:
			j = int((star.x+self.L)//self.DELTA)
			k = int((star.y+self.L)//self.DELTA)
			self.sigma[j,k] += self.MSTAR
		self.sigma /= (self.DELTA * self.DELTA)

	def compute_potential(self):
		for k in range(self.M):
			for l in range(self.M):
				if self.sigma[k,l] > 0.:
					for i in range(self.M+1):
						for j in range(self.M+1):
							dist = np.sqrt((i-k-0.5)**2 + (j-l-0.5)**2)
							self.pot[i,j] += self.sigma[k,l]/dist
		self.pot *= (-self.GGRAV * self.MSTAR * self.DELTA)

	def compute_force(self):
		for star in self.galaxy:
			xs = (star.x % self.DELTA) / self.DELTA
			ys = (star.y % self.DELTA) / self.DELTA
			ix = int((star.x+self.L)/self.DELTA)
			iy = int((star.y+self.L)/self.DELTA)
			star.fx = -((self.pot[ix+1,iy]-self.pot[ix,iy])*(1-ys) + \
						(self.pot[ix+1,iy+1]-self.pot[ix,iy+1])*ys)/self.DELTA
			star.fy = -((self.pot[ix,iy+1]-self.pot[ix,iy])*(1-xs) + \
						(self.pot[ix+1,iy+1]-self.pot[ix+1,iy])*xs)/self.DELTA

	def verlet(self):
		for star in self.galaxy:
			if star.fx0 == 0 and star.fy0 == 0:
				star.fx0 = star.fx
				star.fy0 = star.fy

			star.x += star.vx*self.dt + \
						0.5*star.fx*self.dt*self.dt/self.MSTAR
			star.y += star.vy*self.dt + \
						0.5*star.fy*self.dt*self.dt/self.MSTAR
			star.vx += 0.5 * self.dt * (star.fx0+star.fx)/self.MSTAR
			star.vy += 0.5 * self.dt * (star.fy0+star.fy)/self.MSTAR
			star.fx0, star.fy0 = star.fx, star.fy

			star.adjust_position()

	def show_fig416(self):
		"""Recreation of figure 4.16"""
		radiusplot = []
		deltamplot = []
		mplot = []
		sigmaplot = []
		velocityplot = []
		omegaplot = []

		i=0
		for star in self.galaxy:
			i+=1
			if i % 100000 == 0:
				print(i)
			radiusplot.append(star.radius)
			deltamplot.append(self.deltaM[star.inring]*100)
			mplot.append(self.Mr[star.inring])
			sigmaplot.append(star.area_density)
			velocityplot.append(star.va*3.086e+16/7.5e15)
			omegaplot.append(self.omega[star.inring])

		fig, axarr = plt.subplots(2,sharex=True,figsize=(9,10))
		axarr[0].plot(radiusplot,deltamplot,'ko', ms=0.5)
		axarr[0].plot(radiusplot,mplot,'ko',ms=0.5)
		# Make the y-axis label, ticks and tick labels match the line color.
		axarr[0].set_ylabel(r'$100 \times M(r)/M_{\odot}$, $M(<r)/M_{\odot}$',\
								color='k')
		axarr[0].set_xlim([0,30])
		axarr[0].set_ylim([0,1.1e11])
		axarr[0].text(22,9e10,r'$M(<r)$')
		axarr[0].text(11,5e10,r'$100\times \Delta M(r)$')
		axarr[0].text(22,1e10,r'$N=10^6$')

		ax1 = axarr[0].twinx()
		ax1.plot(radiusplot, sigmaplot, 'ro',ms=0.5)
		ax1.set_ylabel(r'$\sigma (r) (M_{\odot}$ kpc$^{-2}$)', color='r')
		ax1.tick_params('y', which='both',colors='r')
		ax1.set_ylim([0,1e9])
		ax1.set_xlim([0,30])
		ax1.spines['right'].set_color('red')

		axarr[1].plot(radiusplot,velocityplot,'ko',ms=0.5)
		axarr[1].set_ylabel(r'$v_{rot}$ (km s$^{-1}$)', color='k')
		axarr[1].set_xlabel(r'$r$ (kpc)')
		axarr[1].set_xlim([0,30])
		axarr[1].set_ylim([0,250])

		ax2 = axarr[1].twinx()
		ax2.plot(radiusplot,omegaplot,'ro',ms=0.5)
		ax2.set_ylabel(r'$\Omega (r)$', color='r')
		ax2.tick_params('y', which='both',colors='r')
		ax2.set_ylim([0,40])
		ax2.set_xlim([0,30])
		ax2.spines['right'].set_color('red')

		fig.tight_layout()
		fig.savefig("/Users/user/416.png",format='png', dpi=1000)
		plt.show()

	def show_fig417(self,j):
		"""Recreation of figure 4.17"""
		xpos = []
		ypos = []
		for star in self.galaxy:
			xpos.append(star.x)
			ypos.append(star.y)
		plt.figure(figsize=(9,9))
		plt.plot(xpos,ypos,'go',ms=0.25)
		temp = -15
		"""
		for i in range(self.M+1):
			plt.axhline(temp,-15,15,c='r',lw=0.5)
			plt.axvline(temp,-15,15,c='r',lw=0.5)
			temp += self.DELTA
		"""
		plt.axis([-30,30,-30,30])
		#plt.savefig("/Users/user/417.png",format='png', dpi=1000)
		plt.savefig("/Users/user/417_%d.png"%j,format='png')


#-------------------------------Fonctions---------------------------------------

#-------------------------------Main--------------------------------------------

galaxie = Galaxy()
galaxie.compute_mass()
galaxie.init_velocity()
#galaxie.show_fig416()
#galaxie.show_fig417()
galaxie.show_fig417(0)

for i in range(nIter):
	galaxie.compute_density()
	galaxie.compute_potential()
	print(i)
	galaxie.compute_force()
	galaxie.verlet()
	if i%25 == 0:
		galaxie.show_fig417(i+1)

print(time.time()-start)