#！/usr/bin/python
#-*- coding: <encoding name> -*-
import numpy as np
import pandas as pd 
import os
import math

def naca0012(x):
	return 0.6 * (-0.1015 * x ** 4 + 0.2843 * x ** 3 - 0.3576 \
		* x ** 2 - 0.1221 * x + 0.2969 * x ** 0.5)

def simpleIteration(alpha, beta, gamma, x):
	out = (alpha[1:-1, 1:-1] * (x[1:-1, 2:] + x[1:-1, 0:-2]) + \
		gamma[1:-1, 1:-1] * (x[2:, 1:-1] + x[0:-2, 1:-1]) - \
		beta[1:-1, 1:-1] / 2.0 * (x[2:, 2:] + x[0:-2, 0:-2] - x[0:-2, 2:] \
		- x[2:, 0:-2])) / (2.0 * (alpha[1:-1, 1:-1] + gamma[1:-1, 1:-1]))
	return out

def isConverg(dx, dy, errlimit):
	errx = np.abs(dx[1:-1, 1:-1]).max()
	erry = np.abs(dy[1:-1, 1:-1]).max()
	return (errx < errlimit) & (erry < errlimit)

class Ogrid():
	'''
	This is a O-gird generater with:
		- Slitting, far field, airfoil, infield boundary
		- Radius of far field
		- Maximum of iteration and error limits
		- Number of points on the airfoil surface
	'''

	def __init__(self, N, radius, iterlimit, errlimit):
		self.N = N
		self.radius = radius
		self.iterlimit = iterlimit
		self.errlimit = errlimit

	def initialGrid(self):
		n = self.N
		r = self.radius

		theta = np.array([2 * math.pi * (i - 1) / (n - 1) for i in range(0, n + 2)])
		x = np.zeros((n, n + 2))
		y = np.zeros((n, n + 2))

		# Airfoil boundary
		x[0, 1:-1] = 0.5 * (1 + np.cos(theta[1:-1]))
		y[0, 1: int((n + 1) / 2)] = - naca0012(x[0, 1:int((n + 1) / 2)])
		y[0, int((n + 1) / 2):-1] = naca0012(x[0, int((n + 1) / 2):-1])

		# Far field boundary
		x[-1, 1:-1] = r * np.cos(- theta[1:-1])
		y[-1, 1:-1] = r * np.sin(- theta[1:-1])

		# In field boundary
		for i in range(1, n - 1):
			x[i, 1:-1] = x[0, 1:-1] + (x[-1, 1:-1] - x[0, 1:-1]) * i / (n - 1)
			y[i, 1:-1] = y[0, 1:-1] + (y[-1, 1:-1] - y[0, 1:-1]) * i / (n - 1)

		# Slitting (割缝)
		x[:, 0], x[:, -1] = x[:, -3], x[:, 2]
		y[:, 0], y[:, -1] = y[:, -3], y[:, 2]

		return x, y

	def gridGenerate(self):
		# Generate grid using simple iteration
		# Initial variables
		n = self.N

		alpha = np.zeros((n, n+2))
		beta = np.zeros((n, n+2))
		gamma = np.zeros((n, n+2))
		x1 = np.zeros((n, n+2))
		x2 = np.zeros((n, n+2))
		y1 = np.zeros((n, n+2))
		y2 = np.zeros((n, n+2))
		x, y = self.initialGrid()

		for i in range(0, iterlimit):
			xs, ys = x.copy(), y.copy()				# Very important point, with no np.copy(), xs will change with x
			x1[1:-1, 1:-1] = (xs[2:, 1:-1] - xs[0:-2, 1:-1]) / 2.0
			y1[1:-1, 1:-1] = (ys[2:, 1:-1] - ys[0:-2, 1:-1]) / 2.0
			x2[1:-1, 1:-1] = (xs[1:-1, 2:] - xs[1:-1, 0:-2]) / 2.0
			y2[1:-1, 1:-1] = (ys[1:-1, 2:] - ys[1:-1, 0:-2]) / 2.0

			# Calculate alpha, beta and gamma
			alpha[1:-1, 1:-1] = x1[1:-1, 1:-1] ** 2 + y1[1:-1, 1:-1] ** 2
			beta[1:-1, 1:-1] = x2[1:-1, 1:-1] * x1[1:-1, 1:-1] + y2[1:-1, 1:-1] * y1[1:-1, 1:-1]
			gamma[1:-1, 1:-1] = x2[1:-1, 1:-1] ** 2 + y2[1:-1, 1:-1] ** 2

			# Using simple iteration equation
			x[1:-1, 1:-1] = simpleIteration(alpha, beta, gamma, xs)
			y[1:-1, 1:-1] = simpleIteration(alpha, beta, gamma, ys)

			# Slitting 
			x[:, 0], x[:, -1] = x[:, -3], x[:, 2]
			y[:, 0], y[:, -1] = y[:, -3], y[:, 2]

			if isConverg(x - xs, y - ys, self.errlimit):
				return x, y
			elif i == (iterlimit -1):
				return x, y

	def gridExport(self, x, y):
		# Export grid to a dat file
		np.savetxt('Result' + os.sep + 'o-grid.dat', np.transpose([x[:, 1:-1].flatten(), y[:, 1:-1].flatten()]), \
			header='variables=x,y\nzone i=101,j=101,F=POINT', delimiter='\t', comments='')

class GridSolver():
	'''
	This is a o-grid solver with:
		- Grid coordinates input
		- Freestream velocity input
		- Phi, u, v, Cp output
	'''
	def __init__(self, x, y, Vf, errlimit, iterlimit):
		self.x = x
		self.y = y
		self.Vf = Vf
		self.errlimit = errlimit
		self.iterlimit = iterlimit

	def phiSolver(self):
		Vf = self.Vf
		errlimit = self.errlimit
		iterlimit = self.iterlimit
		x = self.x
		y = self.y
		phi, phii, ak = 0*x.copy(), 0*x.copy(), 0*x.copy()

		phi[:, 1:-1] = Vf * x[:, 1:-1]

		for i in range(0, iterlimit):
			phi[:, 0], phi[:, -1] = phi[:, -3], phi[:, 2]
			phis = phi.copy()
			x1 = (x[0, 2] - x[0, 0]) / 2.0
			y1 = (y[0, 2] - y[0, 0]) / 2.0
			x2 = (-x[2, 1] + 4.0 * x[1, 1] - 3.0 * x[0, 1]) / 2.0
			y2 = (-y[2, 1] + 4.0 * y[1, 1] - 3.0 * y[0, 1]) / 2.0
			phi[0, 1] = (4 * phi[1, 1] - phi[2, 1] - (y2 - x2) / (y1 - x1) * (phi[0, 2] - phi[0, 0])) / 3.0
			phi[0, -1] = phi[0, 1]

			x1 = (x[0, 3:-1] - x[0, 1:-3]) / 2.0
			y1 = (y[0, 3:-1] - y[0, 1:-3]) / 2.0
			x2 = (-x[2, 2:-2] + 4.0 * x[1, 2:-2] - 3.0 * x[0, 2:-2]) / 2.0
			y2 = (-y[2, 2:-2] + 4.0 * y[1, 2:-2] - 3.0 * y[0, 2:-2]) / 2.0
			phii[0, 1:-3] = (phi[0, 3:-1] - phi[0, 1:-3]) / 2.0
			ak[0, 1:-3] = (x1 * x2 + y1 * y2) / (x1 ** 2 + y1 ** 2)
			phi[0, 2:-2] = - (2.0 * phii[0, 1:-3] * ak[0, 1:-3] - 4.0 * phi[1, 2:-2] + phi[2, 2:-2]) / 3.0

			phi[1:-1, 0], phi[1:-1, -1] = phi[1:-1, -3], phi[1:-1, 2]
			alpha = ((x[2:, 1:-1] - x[0:-2, 1:-1]) ** 2 + (y[2:, 1:-1] - y[0:-2, 1:-1]) ** 2) / 4.0
			beta = -(((x[2:, 1:-1] - x[0:-2, 1:-1]) * (x[1:-1, 2:] - x[1:-1, 0:-2])) + \
				((y[2:, 1:-1] - y[0:-2, 1:-1]) * (y[1:-1, 2:] - y[1:-1, 0:-2]))) / 4.0
			gamma = ((x[1:-1, 2:] - x[1:-1, 0:-2]) ** 2 + (y[1:-1, 2:] - y[1:-1, 0:-2]) ** 2) / 4.0
			phi[1:-1, 1:-1] = 0.5 * (alpha * (phi[1:-1, 2:] + phi[1:-1, 0:-2]) + gamma * \
				(phi[2:, 1:-1] + phi[0:-2, 1:-1]) + 0.5 * beta * (phi[2:, 2:] + phi[0:-2, 0:-2] -\
					phi[0:-2, 2:] - phi[2:, 0:-2])) / (alpha + gamma)

			if (np.abs((phi - phis)[1:-1,1:-1]).max() < errlimit):
				return phi
			elif i == (iterlimit -1):
				return phi

	def flowSolver(self):
		phi = self.phiSolver()
		x = self.x
		y = self.y
		u, v, Cp = x.copy(), x.copy(), x.copy()

		Jacobi = (x[0, 3:] - x[0, 1:-2]) * (-3.0 * y[0, 2:-1] + 4.0 * y[1, 2:-1] - y[2, 2:-1]) -\
			(y[0, 3:] - y[0, 1:-2]) * (-3.0 * x[0, 2:-1] + 4.0 * x[1, 2:-1] - x[2, 2:-1])
		u[0, 2:-1] = ((phi[0, 3:] - phi[0, 1:-2]) * (-3.0 * y[0, 2:-1] + 4.0 * y[1, 2:-1] - y[2, 2:-1]) -\
			(-3.0 * phi[0, 2:-1] + 4.0 * phi[1, 2:-1] - phi[2, 2:-1]) * (y[0, 3:] - y[0, 1:-2])) / Jacobi
		v[0, 2:-1] = ((x[0, 3:] - x[0, 1:-2]) * (-3.0 * phi[0, 2:-1] + 4.0 * phi[1, 2:-1] - phi[2, 2:-1]) -\
			(-3.0 * x[0, 2:-1] + 4.0 * x[1, 2:-1] - x[2, 2:-1]) * (phi[0, 3:] - phi[0, 1:-2])) / Jacobi
		Cp[0, 2:-1] = 1 - (u[0, 2:-1] ** 2 + v[0, 2:-1] ** 2) / self.Vf ** 2

		u[0, 1] = 0.0
		v[0, 1] = 0.0
		Cp[0, 1] = 1.0
		u[-1, 1:-1] = self.Vf
		v[-1, 1:-1] = 0.0
		Cp[-1, 1:-1] = 0.0

		Jacobi = (x[1:-1, 2:] - x[1:-1, 0:-2]) * (y[2:, 1:-1] - y[0:-2, 1:-1]) - \
			(y[1:-1, 2:] - y[1:-1, 0:-2]) * (x[2:, 1:-1] - x[0:-2, 1:-1])
		u[1:-1, 1:-1] = ((phi[1:-1, 2:] - phi[1:-1, 0:-2]) * (y[2:, 1:-1] - y[0:-2, 1:-1]) -\
			(y[1:-1, 2:] - y[1:-1, 0:-2]) * (phi[2:, 1:-1] - phi[0:-2, 1:-1])) / Jacobi
		v[1:-1, 1:-1] = ((x[1:-1, 2:] - x[1:-1, 0:-2]) * (phi[2:, 1:-1] - phi[0:-2, 1:-1]) -\
			(phi[1:-1, 2:] - phi[1:-1, 0:-2]) * (x[2:, 1:-1] - x[0:-2, 1:-1])) / Jacobi
		Cp[1:-1, 1:-1] = 1 - (u[1:-1, 1:-1] ** 2 + v[1:-1, 1:-1] ** 2) / self.Vf ** 2

		return u[:, 1:-1], v[:, 1:-1], Cp[:, 1:-1], phi[:, 1:-1]

	def flowExport(self, x, y, u, v, Cp):
		# Export flow field variables to a dat file
		np.savetxt('Result' + os.sep + 'flowfield.dat', np.transpose([x[:, 1:-1].flatten(), y[:, 1:-1].flatten(),\
		 u.flatten(), v.flatten(), Cp.flatten()]), fmt='%.10f', \
		header='variables=x,y,u,v,Cp\nzone i=101,j=101,F=POINT', delimiter='\t', comments='')
		np.savetxt('Result' + os.sep + 'Cp-x.dat', np.transpose([x[0, 1:-3].flatten(), Cp[0, 0:-2].flatten()]), \
			header='variables=x,Cp', fmt='%.10f', delimiter='\t', comments='')

if __name__ == '__main__':
	input = pd.read_csv('input.txt', sep=',', usecols=['Points', 'Radius', \
		'iterlimit','errlimit', 'Velocity'])
	N, r, iterlimit, errlimit, Vf = input['Points'][0], input['Radius'][0], \
	input['iterlimit'][0], input['errlimit'][0], input['Velocity'][0]

	ogrid = Ogrid(N, r, iterlimit, errlimit)
	x, y = ogrid.gridGenerate()
	ogrid.gridExport(x, y)
	flow = GridSolver(x, y, Vf, errlimit, iterlimit)
	u, v, Cp, phi = flow.flowSolver()
	flow.flowExport(x, y, u, v, Cp)