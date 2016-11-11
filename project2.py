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
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5)

def gauss(A, b, x, alpha, epsilon = 1e-5):
	'Run n iterations to solve Ax=b using Gauss-Sidel'
	# thanks http://austingwalters.com/gauss-seidel-method/
	L = np.tril(A)
	U = A - L

	res = epsilon + 1
	while res > epsilon:
		x_star = x # initial guess
		x = np.dot(np.linalg.inv(L), b - np.dot(U, x_star))
		x = x*alpha + x_star*(1-alpha) # under relaxation
		res = residual(x, x_star)
		print("res="+str(res), end='\r')

	return x

def residual(a,b):
	return np.sum(np.absolute(a - b))

def solve_lamda(eta, phi):
	lamda = 1./(4. * np.trapz(phi*eta*(1-np.square(eta)), eta))
	return lamda

def sourceTerm(eta, lamda, phi):
	'Linearize the source term S into S_C + S_P*phi'
	S_C = lamda * (1-np.square(eta)) * phi
	S_P = np.full(len(eta), 0, float)
	return S_C, S_P

def coefficients(eta, d_eta, S_C, S_P, Bi):
	'Returns the coefficients to solve a_P*T_P = a_E*T_E + a_W*T_W + b'
	N = len(eta)
	A = np.zeros(N**2).reshape(N, N)
	b = np.zeros(N)

	for i in xrange(len(eta)):
		# A[i][i] is like aP
		# A[i][i-1] is like aW
		# A[i][i++] is like aE
		if i == 0:
			#symmetrical centerline
			dV = (d_eta/2)**2
			# 2nd order
			A[i][i] = -S_P[i]*dV + 1e-9 
			A[i][i+1] = 0 # 2nd order

			# 1st order
			#A[i][i] = (1/d_eta+0.5 - S_P[i]*dV) 
			#A[i][i+1] = -(1/d_eta+0.5)

			b[i] = S_C[i]*dV
		elif i == N-1:
			#wall flux
			dV = eta[i]*d_eta
			#first order, given by Prof Yan:
			#A[i][i] = eta[i]/d_eta - (2. - d_eta**2*Bi)
			#A[i][i-1] = 2/d_eta
			#b[i] = 0

			# 2nd order
			A[i][i] = Bi*(d_eta+d_eta**2/2.) - 1. - S_P[i]*dV
			A[i][i-1] = -1

			# 1st order
			#A[i][i] = (1./d_eta - 0.5 - S_P[i]*dV + Bi*(d_eta/2. - 1./d_eta))
			#A[i][i-1] = -(1./d_eta-0.5)

			b[i] = S_C[i]*dV
		else:
			dV = eta[i]*d_eta
			A[i][i] = 1*(eta[i]/d_eta + 0.5) + 1*(eta[i]/d_eta - 0.5) - S_P[i]*dV
			A[i][i-1] = -1*(eta[i]/d_eta + 0.5) # I know the +0.5 / -0.5 are swapped
			A[i][i+1] = -1*(eta[i]/d_eta - 0.5) # but this gives the right shape to the result
			b[i] = S_C[i]*dV
	return A, b

def symmBC(nb):
	return nb

def fluxBC(nb, d_eta, Bi):
	return nb/(1.0 + d_eta*Bi)


def solvePhi(N, eta, Bi, epsilon, debug):

	d_eta = eta[1]-eta[0]
	phi = np.ones(N)
	print("NEW RUN, Bi="+str(Bi))
	if debug:
		print("eta="+str(eta))
		print("d_eta="+str(d_eta))

	alpha = .9
	res = epsilon + 1
	while res > epsilon:
		phi_star = phi
		lamda = solve_lamda(eta, phi_star)
		S_C, S_P = sourceTerm(eta, lamda, phi_star)
		
		A, b = coefficients(eta, d_eta, S_C, S_P, Bi)
		if debug:
			print("S_C="+str(S_C))
			print("S_P="+str(S_P))
			print("A="+str(A))
			#print("phi*="+str(phi_star))
			print("b="+str(b))
			print("lamda="+str(lamda))

		
		phi = gauss(A, b, phi_star, alpha, epsilon)
		phi[0] = symmBC(phi[1])
		phi[-1] = fluxBC(phi[-2], d_eta, Bi)

		res = residual(phi, phi_star)
		
		if debug:
			print("theta="+str(lamda*phi))
		print("res="+str(res), end='\r')

	theta = lamda*phi
	Nu = 2.*theta[-1]/(1.-theta[-1])*Bi

	if debug:
		print("theta="+str(theta))

	print("Bi="+str(Bi)+", lamda="+str(lamda)+", Nu="+str(Nu))

	label = "Bi="+str(Bi)+"\nNu="+str(np.round(Nu,3))
	plt.plot(eta, theta, label=label)
	return phi



if __name__ == '__main__':
	#setup grid
	N = 90
	eta = np.linspace(0, 1, N, True)
	
	plt.figure(1)
	plt.clf()
	plt.xlabel("$Nondimensional Radial Position - \eta$")
	plt.ylabel("$Nondimensional Temperature - \\theta$")

	debug = 0
	
	Bis = [0., 0.5, 1., 5., 10., 1.0e6]
	for Bi in Bis:
		solvePhi(N, eta, Bi, 1e-5, debug)
	
	sp=plt.subplot(1, 1, 1)
	box=sp.get_position()
	sp.set_position([box.x0, box.y0, box.width*0.8, box.height])
	plt.legend(bbox_to_anchor=(1, 0.5), loc='center left')
	plt.show()