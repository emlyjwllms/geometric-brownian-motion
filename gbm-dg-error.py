# DG geometric Brownian motion error

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
    xdot = (alpha - 0.5*beta**2)*xv + beta*xv*dW[j]/Dt
    return xdot


# residual vector
def resid(c,xh0):
    r = np.zeros((porder+1))
    for k in range(porder+1):
        phi_k = phi[:,k] # phi_k (xi_q)
        dphi_k = dphi[:,k] # phi'_k (xi_q)
        phi_k_left = phi_left[:,k] # phi_k (xi_q = -1)
        phi_k_right = phi_right[:,k] # phi_k (xi_q = +1)

        r[k] = dphi_k.T @ np.diag(w) @ phi @ c + phi_k.T @ np.diag(0.5*Dt*w) @ f(phi @ c) - phi_k_right.T @ phi_right @ c + phi_k_left.T @ np.array([xh0])

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

    # params
    alpha = 2
    beta = 0.5
    x0 = 1

    mc = 1000

    dt_grid = []

    for R in np.arange(0,5,2, dtype=float):
            dt_grid = np.append(dt_grid,2**(R-10))

    str_err_dg = []
    weak_err_dg = []

    porders = np.array([1])

    for pi in range(len(porders)):

        porder = porders[pi]

        # quadrature points
        xi, w = gaussquad1d(porder+1)
        Nq = len(xi)

        # precompute polynomials
        phi, dphi = plegendre(xi,porder)
        phi_left, dphi_left = plegendre(-1,porder) # dphi_left not used
        phi_right, dphi_right = plegendre(1,porder) # dphi_right not used

        for Dt_i in range(len(dt_grid)):
            Dt = dt_grid[Dt_i]
            t = np.arange(Dt,1+Dt,Dt)
            n = len(t)

            err_dg = 0
            x_exact_sum = 0
            x_dg_sum = 0

            for i in range(mc):
                np.random.seed(i)
                #dW = np.sqrt(Dt) * np.random.normal(0,np.sqrt(Dt),size=n)
                dW = np.sqrt(Dt) * np.random.randn(n)
                W = np.cumsum(dW)
                x_exact = x0 * np.exp(beta*W + (alpha - 0.5*beta**2)*t)

                # initial conditions
                xh = np.zeros((1)) # elements x quad points X x
                xhq = np.zeros((1)) # all quad points
                tq = np.zeros((1)) # time points that correspond to quad points
                xh[0] = x0
                xhq = xh[0]
                xh0 = xh[0]
                cguess = np.append(xh0,np.zeros(porder))

                # integrate across time elements - DG
                for j in range(1,n): # loop across I_j's
                    t0 = t[j-1]
                    tf = t[j]
                    c = scipy.optimize.root(resid, cguess, args=(xh0,)).x # solve residual function above
                    xhq = np.append(xhq,phi @ c)
                    tq = np.append(tq,Dt*xi/2 + (t0+tf)/2)
                    xh = np.append(xh,phi_right @ c) # xi = +1
                    cguess = c
                    xh0 = xh[-1]

                err_dg += np.abs(x_exact - xh)

                x_exact_sum += x_exact
                x_dg_sum += xh

            str_err_dg = np.append(str_err_dg,np.max(err_dg/mc))

            weak_err_dg = np.append(weak_err_dg,np.max(np.abs(x_exact_sum - x_dg_sum)/mc))

    str_err_dg = np.reshape(str_err_dg,(len(porders),len(dt_grid)))
    weak_err_dg = np.reshape(weak_err_dg,(len(porders),len(dt_grid)))

    np.savez('gbm_dg_err', dt_grid=dt_grid, str_err_dg=str_err_dg, weak_err_dg=weak_err_dg )
