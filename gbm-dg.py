# 1D DG solver for geometric Brownian motion - one realization

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from scipy.special import jacobi

# xdot function
def f(xv):
    # dX = a X dt + beta X dW
    # xdot = a X + beta X dW/dt
    xdot = (alpha - 0.5*beta**2)*xv + beta*xv*dW[j]/dt
    return xdot


# residual vector
def resid(c,xh0):
    r = np.zeros((porder+1))
    for k in range(porder+1):
        phi_k = phi[:,k] # phi_k (xi_q)
        dphi_k = dphi[:,k] # phi'_k (xi_q)
        phi_k_left = phi_left[:,k] # phi_k (xi_q = -1)
        phi_k_right = phi_right[:,k] # phi_k (xi_q = +1)

        r[k] = dphi_k.T @ np.diag(w) @ phi @ c + phi_k.T @ np.diag(0.5*dt*w) @ f(phi @ c) - phi_k_right.T @ phi_right @ c + phi_k_left.T @ np.array([xh0])

    return r

def gaussquad1d(pgauss):

    """     
    gaussquad1d calculates the gauss integration points in 1d for [-1,1]
    [x,w]=gaussquad1d(pgauss)

      x:         coordinates of the integration points 
      w:         weights  
      pgauss:         order of the polynomila integrated exactly 
    """

    n = math.ceil((pgauss+1)/2)
    P = jacobi(n, 0, 0)
    x = np.sort(np.roots(P))

    A = np.zeros((n,n))
    for i in range(1,n+1):
        P = jacobi(i-1,0,0)
        A[i-1,:] = np.polyval(P,x)

    r = np.zeros((n,), dtype=float)
    r[0] = 2.0
    w = np.linalg.solve(A,r)

    # map from [-1,1] to [0,1]
    #x = (x + 1.0)/2.0
    #w = w/2.0

    return x, w

def plegendre(x,porder):
    
    try:
        y = np.zeros((len(x),porder+1))
        dy = np.zeros((len(x),porder+1))
        ddy = np.zeros((len(x),porder+1))
    except TypeError: # if passing in single x-point
        y = np.zeros((1,porder+1))
        dy = np.zeros((1,porder+1))
        ddy = np.zeros((1,porder+1))

    y[:,0] = 1
    dy[:,0] = 0
    ddy[:,0] = 0

    if porder >= 1:
        y[:,1] = x
        dy[:,1] = 1
        ddy[:,1] = 0
    
    for i in np.arange(1,porder):
        y[:,i+1] = ((2*i+1)*x*y[:,i]-i*y[:,i-1])/(i+1)
        dy[:,i+1] = ((2*i+1)*x*dy[:,i]+(2*i+1)*y[:,i]-i*dy[:,i-1])/(i+1)
        ddy[:,i+1] = ((2*i+1)*x*ddy[:,i]+2*(2*i+1)*dy[:,i]-i*ddy[:,i-1])/(i+1)

    # return y,dy,ddy
    return y,dy


if __name__ == "__main__":

    porder = 1

    # params
    alpha = 2
    beta = 0.5
    x0 = 1

    # simulation parameters
    T = 1
    N = 2**9
    dt = 1/N
    t = np.arange(0,T,dt)

    # quadrature points
    xi, w = gaussquad1d(porder+1)
    Nq = len(xi)

    # initial conditions
    xh = np.zeros((1)) # elements x quad points X x
    xhq = np.zeros((1)) # all quad points
    tq = np.zeros((1)) # time points that correspond to quad points
    xh[0] = x0
    xhq = xh[0]

    np.random.seed(1)
    dW = np.sqrt(dt) * np.random.randn(N)
    W = np.cumsum(dW)
    x_exact = x0 * np.exp(beta*W + (alpha - 0.5*beta**2)*t)

    # precompute polynomials
    phi, dphi = plegendre(xi,porder)
    phi_left, dphi_left = plegendre(-1,porder) # dphi_left not used
    phi_right, dphi_right = plegendre(1,porder) # dphi_right not used

    xh0 = xh[0]
    cguess = np.append(xh0,np.zeros(porder))
    
    # integrate across time elements - DG
    for j in range(1,N): # loop across I_j's
        t0 = t[j-1]
        tf = t[j]
        c = scipy.optimize.root(resid, cguess, args=(xh0,)).x # solve residual function above
        xhq = np.append(xhq,phi @ c)
        tq = np.append(tq,dt*xi/2 + (t0+tf)/2)
        xh = np.append(xh,phi_right @ c) # xi = +1
        cguess = c
        xh0 = xh[-1]
        

    plt.figure(figsize=(4,6))
    plt.plot(t,x_exact,'k-',linewidth=2,zorder=1,label="Analytical")
    #plt.scatter(tq,xhq,s=5,zorder=2)
    #plt.scatter(t,xh,s=5,zorder=2)
    plt.plot(t,xh,zorder=1,label="DG")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.grid()
    plt.legend()

    plt.savefig('plots/gbm-dg.png',dpi=300,format='png',transparent=True,bbox_inches='tight')


    plt.show()


