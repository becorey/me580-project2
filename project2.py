# Author : Corey Berman
# ME 580, Fall 2016, Prof Terry Yan
# 1D heat transfer
# hydrodynammically fully developed flow
# in a round pipe

# Procedure:
#  1) Initial guess phi_star
#  2) Solve for lamda_star = 
#      1/(4*integral(phi*eta*(1-eta^2)d_eta, 0,1))
#  3) With lamda_star, solve the ODE
#      lamda*(1/eta)*(d/d_eta)*(eta*d_phi/d_eta) + 
#        lamda^2*(1-eta^2)*phi = 0
#     Iterate until |phi-phi_star| <= epsilon
#  4) theta = lamda * phi
#     theta(r=R) = theta_wall
#     Nu = 2*theta_wall/(1-theta_wall)*Bi

import numpy as np


def SOR(A, b, alpha = .66, epsilon = 1e-4):
	'Solves Ax=b for x using Gauss-Sidel Successive Over Relaxation'
	residual = 1
	while residual > epsilon:
		for i in range(0, len(A)):
			var = 1


def solve_lamda(eta, phi):
	print "running solve_lamda"
	y = phi*eta*(1-np.square(eta))
	print "y="+str(y)
	print "eta="+str(eta)
	lamda = 1./4.*1./np.trapz(y, eta)
	print "lamda="+str(lamda)
	return lamda

def solve_phi(eta, lamda, phi, epsilon = 1e-4):
	phi_star = phi
	residual = 1
	while residual > epsilon:
		aP = np.square(lamda)*(1-np.square(eta)) - 2*lamda/np.square(d_eta)
		aE = np.full(len(eta), -lamda/(4*d_eta) - lamda/np.square(d_eta))
		aW = np.full(len(eta), lamda/(4*d_eta) - lamda/np.square(d_eta))
		print aP
		print aE
		print aW
		for i in range(len(phi)):
			if i == 0:
				# symmetrical BC
				phi_star[i] = 5.
			elif i == len(phi)-1:
				# wall BC
				phi_star[i] = 10.
			else:
				#phi[i] = (aE[i]*phi[i+1] + aW[i]*phi[i-1] + b[i])/aP[i]
				# GS method, use latest (phi_star) for already calculated [i-1]
				phi_star[i] = aE[i]/aP[i]*phi[i+1] + aW[i]/aP[i]*phi_star[i-1]
		print phi-phi_star
		residual = np.sum(np.absolute(phi - phi_star))
		print "residual="+str(residual)
		phi = phi_star
	return phi

if __name__ == '__main__':
	N = 4
	eta = np.linspace(0, 1, N, True)
	d_eta = eta[1]-eta[0]
	phi = np.ones(N)
	lamda = solve_lamda(eta, phi)
	phi = solve_phi(eta, lamda, phi)

