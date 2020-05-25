#！/usr/bin/python
#-*- coding: <encoding name> -*-
import numpy as np
import pandas as pd 
import os
import math

def initCondition(xl, xr, N, pe, Ma):
	dx = (xr - xl) / N
	x = np.array([xl + dx * i for i in range(0, N + 1)])
	Ax = 0.5 + x ** 2
	# Initiate the u, p, rho
	u = 2 * np.ones(N + 1)
	p = 2 * np.ones(N + 1)
	rho = 2 * np.ones(N + 1)

	u[0] = Ma * math.sqrt(1.4 / (1 + 0.2 * Ma ** 2))
	u[-1] = 2 * u[-2] - u[-3]
	p[0] = 1 / (1 + 0.2 * Ma ** 2) ** (1.4 / 0.4)
	p[-1] = pe
	rho[0] = 1 / (1 + 0.2 * Ma ** 2) ** (1 / 0.4)
	rho[-1] = rho[-2] * (p[-1] / p[-2]) ** (1 / 1.4)

	return u, p, rho, x, Ax, dx


def calTimeStep(dx, CFL, u, a):
	dt = np.zeros(len(u))
	dt = CFL * dx / (a + np.abs(u))
	return dt.min()

def energy(u, p, rho):
	return 1 / 0.4 * p / rho + 0.5 * u ** 2

def deEnergy(U):
	return 0.4 * (U[2] - 0.5 * U[1] ** 2 / U[0])

def isConverg(dU_dtav, errlimit):
	err = np.abs(dU_dtav).max()
	return (err < errlimit)

class Laval():
	'''
	A laval quasi-one-dimension flow problem solver with:
		- Artificial viscosity and macCormack method
		- Cx = 0.05, 0.15, 0.20
		- Adaptable input of CFL (Courant-Friedrichs-Lowry) and Cx
	Initial and boundary conditions are:
		- Inlet: p0=1, rho0=1, inlet Mach = 1.16
		- Outlet: pe=0.8785
		- Area distribution: A(x)=0.5+x^2, x belongs to [0.1,1.0]
	'''

	def __init__(self, Cx, CFL, dx, x, Ax):
		# Give the artificial viscosity coefficient, CFL and gird spacing.
		# Give discretization of x and Ax
		self.Cx = Cx
		self.CFL = CFL
		self.dx = dx
		self.x = x
		self.Ax = Ax

	def Couples(self, u, p, rho):
		# Couples u, p, rho to U, F, H
		E = energy(u, p, rho)
		# Denote F, H using U
		U = np.array([rho, rho * u, rho * E])
		F = np.array([U[1], U[1] ** 2 / U[0] + deEnergy(U), \
			(U[2] + deEnergy(U)) * U[1] / U[0]])
		H = np.array([2 * self.x * U[1] / self.Ax, \
			2 * self.x * (U[1] ** 2 / U[0]) / self.Ax, \
			2 * self.x * ((U[2] + deEnergy(U)) * U[1] / U[0]) / self.Ax])

		return U, F, H

	def deCouples(self, U):
		# Decouples the U to rho, u, p
		rho = U[0]
		u = U[1] / rho
		p = 0.4 * (U[2] - 0.5 * rho * u ** 2)

		return u, p, rho

	def macCormack(self, u, p, rho, errlimit, iterlimit):
		'''
		u, p, rho are numpy arrays with initial condition.
			- Add artificial viscosity coefficient
			- Classical macCormack method
		'''
		dx = self.dx
		CFL = self.CFL
		Cx = self.Cx

		for i in range(0, iterlimit):
			# Define minimum time step
			a = np.sqrt(1.4 * p / rho)
			dt = calTimeStep(dx, CFL, u, a)

			# Predictive Step
			U, F, H = self.Couples(u, p, rho)	# Couple u, p, rho to U, F, H

			dU_dt = np.zeros((3, len(u)))
			S = np.zeros((3, len(u)))

			dU_dt[:, 1:-1] = -((F[:, 2:] - F[:, 1:-1]) / dx + H[:, 1:-1])
			S[:, 1:-1] = Cx * np.abs(p[2:] - 2 * p[1:-1] + p[0:-2]) / \
			(p[2:] + 2 * p[1:-1] + p[0:-2]) * \
			(U[:, 2:] - 2 * U[:, 1:-1] + U[:, 0:-2])	# Artificial Viscosity Add
			Us = U + dU_dt * dt + S

			# Decouple U, F, H to us, ps, rhos, and suiting outlet boundary conditions
			us, ps, rhos = self.deCouples(Us)
			us[-1] = 2 * us[-2] - us[-3]
			rhos[-1] = rhos[-2] * (ps[-1] / ps[-2]) ** (1 / 1.4)

			# Correction Step
			Us, F, H = self.Couples(us, ps, rhos)	# Couple us, ps, rhos to Us, F, H

			dUs_dt = np.zeros((3, len(u)))
			dU_dtav = np.zeros((3, len(u)))
			Ss = np.zeros((3, len(u)))

			dUs_dt[:, 1:-1] = -((F[:, 1:-1] - F[:, 0:-2]) / dx + H[:, 1:-1])
			dU_dtav = 0.5 * (dUs_dt + dU_dt)
			Ss[:, 1:-1] = Cx * np.abs(ps[2:] - 2 * ps[1:-1] + ps[0:-2]) / \
			(ps[2:] + 2 * ps[1:-1] + ps[0:-2]) * \
			(Us[:, 2:] - 2 * Us[:, 1:-1] + Us[:, 0:-2])		# Add artificial viscosity again
			Ut = U + dU_dtav * dt + Ss

			# Decouple the Ut to ut, pt, rhot(the value in next time step), and suiting outlet boundary conditions
			ut, pt, rhot = self.deCouples(Ut)
			ut[-1] = 2 * ut[-2] - ut[-3]
			rhot[-1] = rhot[-2] * (pt[-1] / pt[-2]) ** (1 / 1.4)

			u, p, rho = ut, pt, rhot

			if isConverg(dU_dtav[1:-1], errlimit):
				return u, p, rho
			elif i == (iterlimit - 1):
				return u, p, rho

if __name__ == '__main__':
	sep = os.sep
	input = pd.read_csv('input.txt', sep = ',', usecols=['Mach Number', 'xleft', 'xright', \
	'Pe', 'Cx', 'CFL', 'Errlimit', 'Iterlimit', 'xStep'])	# Read csv file as input. Attention! inputs are lists.
	Ma, xl, xr, pe, Cx, CFL, errlimit, iterlimit, N = (input['Mach Number'][0], input['xleft'][0], \
		input['xright'][0], input['Pe'][0], input['Cx'][0], input['CFL'][0], input['Errlimit'][0], \
		input['Iterlimit'][0], input['xStep'][0])

	u, p, rho, x, Ax, dx = initCondition(xl, xr, N, pe, Ma)
	laval = Laval(Cx, CFL, dx, x, Ax)
	u, p, rho = laval.macCormack(u, p, rho, errlimit, iterlimit)
	Ma = u / (np.sqrt(1.4 * p / rho))
	flux = rho * u * Ax

	output = pd.DataFrame({'X/L':x, 'Mach': Ma, 'Velocity':u, 'Pressure':p, \
		'Density':rho, 'Flux':flux})
	output.to_csv('Result' + sep + 'Cx=' + str(Cx) + '.dat', sep='\t', index=False)