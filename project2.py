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
#        lamda*(1-eta^2)*phi = 0
#  4) Iterate until |phi-phi_star| <= epsilon
#  5) theta = lamda * phi
#     theta(r=R) = theta_wall
#     Nu = 2*theta_wall/(1-theta_wall)*Bi

import numpy as np
import matplotlib.pyplot as plt

def gauss(A, b, x, alpha, epsilon = 1e-5):
	'Run n iterations to solve Ax=b using Gauss-Sidel'
	# thanks http://austingwalters.com/gauss-seidel-method/
	L = np.tril(A)
	U = A - L

	res = epsilon + 1
	while res > epsilon:
		x_star = x # initial guess
		x = np.dot(np.linalg.inv(L), b - np.dot(U, x_star))
		x = x*alpha + x_star*(1-alpha)
		res = residual(x, x_star)

	return x

def residual(a,b):
	return np.sum(np.absolute(a - b))

def solve_lamda(eta, phi):
	y = phi*eta*(1-np.square(eta))
	lamda = 1./4.*1./np.trapz(y, eta)
	#print "y="+str(y)
	return 1*lamda

def sourceTerm(eta, lamda, phi):
	'Linearize the source term S into S_C + S_P*theta'
	#S_C = np.full(len(eta), 0, float)
	S_C = lamda * (1-np.square(eta)) * phi
	S_P = np.full(len(eta), 0, float)
	return S_C, S_P

def coefficients(eta, d_eta, S_C, S_P):
	'Returns the coefficients to solve a_P*T_P = a_E*T_E + a_W*T_W + b'
	N = len(eta) # exclude first and last point BCs
	A = np.zeros(N**2).reshape(N, N)
	b = np.zeros(N)

	for i in range(len(eta)):
		# A[i][i] is like aP
		# A[i][i-1] is like aW
		# A[i][i++] is like aE
		if i == 0:
			#symmetrical centerline
			A[i][i+1] = 0
			A[i][i] = (d_eta/2)**2*S_P[i] + 1e-6 #needs to be nonzero to avoid singular matrix error
			b[i] = S_C[i]*(d_eta/2)**2
		elif i >= N-1:
			#wall flux
			#first order, given by Prof:
			#A[i][i] = 1/d_eta - (2-d_eta**2*Bi)
			#A[i][i-1] = -2/d_eta
			#b[i] = 0
			#first order, I calculated:
			A[i][i] = (1/d_eta - 0.5 + 2*Bi - S_P[i]*d_eta**2*Bi - S_P[i]*d_eta)
			A[i][i-1] = 0
			A[i][i-2] = -(1/d_eta - 0.5)
			b[i] = S_C[i]*eta[i]*d_eta
			#second order, I calculated:
			#A[i][i] = (-1/d_eta + d_eta*Bi*(d_eta/2+1) - 1 - S_P[i]*d_eta)
			#A[i][i-1] = 1
			#b[i] = S_C[i]*d_eta
		else:
			A[i][i] = 1*(eta[i]/d_eta + 0.5) + 1*(eta[i]/d_eta - 0.5) - S_P[i]*eta[i]*d_eta
			A[i][i-1] = -1*(eta[i]/d_eta - 0.5)
			A[i][i+1] = -1*(eta[i]/d_eta + 0.5)
			b[i] = S_C[i]*eta[i]*d_eta
	return A, b

def symmBC(nb):
	return nb

def fluxBC(nb, d_eta, Bi):
	return nb/(1.0 + d_eta*Bi)


if __name__ == '__main__':
	#setup grid
	N = 80
	eta = np.linspace(0, 1, N, True)
	d_eta = eta[1]-eta[0]
	print "NEW RUN"
	print "eta="+str(eta)
	print "d_eta="+str(d_eta)

	Bi = 1
	phi = np.ones(N)

	debug = 0

	epsilon = 1e-4
	res = epsilon + 1
	while res > epsilon:
		phi_star = phi
		lamda = solve_lamda(eta, phi_star)
		S_C, S_P = sourceTerm(eta, lamda, phi_star)
		
		A, b = coefficients(eta, d_eta, S_C, S_P)
		if debug:
			print "S_C="+str(S_C)
			print "S_P="+str(S_P)
			print "A="+str(A)
			print "phi*="+str(phi_star)
			print "b="+str(b)
			print "lamda="+str(lamda)

		
		phi = gauss(A, b, phi_star, .6, 1e-5)
		phi[0] = symmBC(phi[1])
		phi[-1] = fluxBC(phi[-2], d_eta, Bi)

		res = residual(phi, phi_star)
		
		if debug:
			print "theta="+str(lamda*phi)
		print "res="+str(res)

	theta = lamda*phi
	
	print "lamda="+str(lamda) 
	print "S_C="+str(S_C)
	print "S_P="+str(S_P)
	print "A="+str(A)
	print "phi="+str(phi)
	print "b="+str(b)
	
	print "theta="+str(theta)
	
	plt.figure(1)
	plt.clf()
	plt.plot(eta, theta)
	plt.show()
