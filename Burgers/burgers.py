#！/usr/bin/python
#-*- coding: <encoding name> -*-
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

class Burgers:
	'''
	A Burgers 1-demension partial differential equation numerical solver with:
		- 3 inputs: viscosity ,time and iteration times N
		- viscosity = 0, 0.01, 0.05
		- time = 0.5, 1.2, 2.0
	Initial and boundary conditions are:
		- u(x,0) = -0.5*x, x belongs to [-1, 1]
	    - u(-1, t) = 0.5, u(1, t) = -0.5
	'''

	def __init__(self, visco, time, N):
		# Give the viscosity, time to stop and gird spacing.
	    self.visco = visco
	    self.time = time
	    self.N = N

	def discretization(self):
		# Discretization of the calculation zone.
		dx = 2.0 / self.N
		dt = 0.0001
		return (dx, dt)

	def macCormack(self, u):
		'''
		u is a numpy array with initial condition.
			- Predictive step: us;
			- Patial of test step: up
			- Intermediate step: unew
			- Storage of all the time step: U
		'''
		dx, dt = self.discretization()
		nt = int(self.time / dt)

		us = np.zeros(self.N + 1)
		up = np.zeros(self.N + 1)
		unew = np.zeros(self.N + 1)
		U = np.zeros((nt + 1, self.N + 1))
		U[0] = u

		for i in range(1, nt + 1):
			# Predictive step
			us[1: -1] = u[1: -1] + (self.visco * dt / (dx) ** 2) * (u[2:] - 2.0 * u[1:-1] + u[0:-2]) \
			- dt / (2.0 * dx) * (u[2:] ** 2 - u[1:-1] ** 2)
			us[0] = 0.5
			us[-1] = -0.5 

			# Correction step and combine step
			up[1: -1] = (self.visco / (dx) ** 2) * (us[2:] - 2 * us[1: -1] + us[0: -2]) \
			- 1.0 / (2.0 * dx) * (us[1:-1] ** 2 - us[0: -2] ** 2)
			unew[1: -1] = 0.5 * (u[1: -1] + us[1: -1] + dt * up[1: -1])
			u[1: -1] = unew[1: -1]

			U[i] = u

		return U

if __name__ == '__main__':
	N = 200
	seper = os.sep 		# Use this seperator to satisfy both Windows and Linux usage

	# Read input file of time and viscosity
	time_visco = pd.read_csv('input.txt', sep = ',', usecols=['time', 'mu'])
	time = max(time_visco['time'])
	visco = time_visco['mu']

	# Compute Burgers equation at different viscosity
	for i in range(0, len(visco)):
		u = np.zeros(N + 1)
		x = np.zeros(N + 1)

		for j in range(0, N + 1):
			u[j] = -0.5 * (-1 + j * 2 / N)
			x[j] = -1 + j * 2 / N

		burgers = Burgers(visco[i], time, N)
		uall = burgers.macCormack(u)

		# Save 2-D data to files
		output = pd.DataFrame({'x': x, 't='+str(time_visco['time'][0]): uall[int(time_visco['time'][0] * 10000)], \
			't='+str(time_visco['time'][1]): uall[int(time_visco['time'][1] * 10000)], \
			't='+str(time_visco['time'][2]): uall[int(time_visco['time'][2] * 10000)]})
		output.to_csv('2D' + seper + 'mu=' + str(visco[i]) + '.dat', sep='\t', index=False)

		'''
			Plot 3-D Burgers results, this cost about 16 s, so I comment code below.
			If you want to generate 3-D images by yourself, please uncomment the code below, 
			and change some value of the code, to see if there are some differences between
			my output 3-D images
		'''

		'''
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		y = np.array([[i * 0.0001] for i in range(0, len(uall))])
		ax.plot_surface(x, y, uall, cmap='viridis')
		ax.set_xlabel('x')
		ax.set_ylabel('t')
		ax.set_zlabel('u')
		ax.set_title(r'$\mu = $' + str(visco[i]))
		plt.savefig('3D' + seper + 'mu=' + str(visco[i]) + '.png', bbox_inches='tight')
		'''