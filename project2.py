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
#  4) Iterate until |phi-phi_star| <= epsilon
#  5) theta = lamda * phi
#     theta(r=R) = theta_wall
#     Nu = 2*theta_wall/(1-theta_wall)*Bi

import numpy as np
import matplotlib.pyplot as plt

def gauss(A, b, x, epsilon = 1e-4):
	'Run n iterations to solve Ax=b using Gauss-Sidel'
	# thanks http://austingwalters.com/gauss-seidel-method/
	L = np.tril(A)
	U = A - L

	res = epsilon + 1
	while res > epsilon:
		x_star = x # initial guess
		x = np.dot(np.linalg.inv(L), b - np.dot(U, x_star))
		res = residual(x, x_star)

	return x

def residual(a,b):
	return np.sum(np.absolute(a - b))

def sourceTerm(eta, lamda):
	'Linearize the source term S into S_C + S_P*theta'
	S_C = np.full(len(eta), 0, float)
	S_P = lamda*(1-np.square(eta))
	return S_C, S_P

def coefficients(eta, d_eta, Bi, S_C, S_P):
	'Returns the coefficients to solve a_P*T_P = a_E*T_E + a_W*T_W + b'
	N = len(eta)
	A = np.zeros(N**2).reshape(N, N)
	b = np.zeros(N)

	for i in range(len(eta)):
		# A[i][i] is like aP
		# A[i][i-1] is like aW
		# A[i][i++] is like aE
		if i == 0:
			# symmetric BC
			A[i][i+1] = 2*(eta[i]+d_eta/2)/d_eta # 2* because aW=aE and TW=TE
			A[i][i] = -1*(A[i][i+1] - S_P[i]*eta[i]*d_eta/2)
			b[i] = S_C[i]*eta[i]*d_eta/2
		elif i == N-1:
			# wall heat flux BC
			A[i][i-1] = 1*(eta[i]-d_eta/2)/d_eta
			A[i][i] = -1*(A[i][i-1] - S_P[i]*eta[i]*d_eta/2 + Bi)
			b[i] = S_C[i]*eta[i]*d_eta/2 + 0 # 0 for h_e*T_inf ... when its nondimensionalized, T_inf => theta = 0. not sure if this is correct
		else:
			A[i][i-1] = 1*(eta[i]-d_eta/2)/d_eta
			A[i][i+1] = 1*(eta[i]+d_eta/2)/d_eta
			A[i][i] = -1*(A[i][i-1] + A[i][i+1] - S_P[i]*eta[i]*d_eta/2)
			b[i] = S_C[i]*eta[i]*d_eta/2
	return A, b

def solve_lamda(eta, phi):
	y = phi*eta*(1-np.square(eta))
	lamda = 1./4.*1./np.trapz(y, eta)
	print "solving lamda.."
	print "y="+str(y)
	print "lamda="+str(lamda)
	return -1*lamda

if __name__ == '__main__':
	#setup grid
	N = 50
	eta = np.linspace(0, 1, N, True)
	d_eta = eta[1]-eta[0]
	print "NEW RUN"
	print "eta="+str(eta)
	print "d_eta="+str(d_eta)

	Bi = 1
	phi = np.ones(N)

	epsilon = 1e-5
	res = epsilon + 1
	while res > epsilon:
		phi_star = phi
		lamda = solve_lamda(eta, phi_star)
		S_C, S_P = sourceTerm(eta, lamda)
		print "S_C="+str(S_C)
		print "S_P="+str(S_P)
		A, b = coefficients(eta, d_eta, Bi, S_C, S_P)
		print "A="+str(A)
		print "b="+str(b)
		phi = gauss(A, b, phi_star, 1e-5)

		res = residual(phi, phi_star)
		print "theta="+str(lamda*phi)
		print "res="+str(res)

	theta = lamda*phi
	#print theta
	
	plt.figure(1)
	plt.clf()
	plt.plot(eta, theta)
	plt.show()
