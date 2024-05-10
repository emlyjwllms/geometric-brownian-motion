# geometric Brownian motion Monte Carlo simulation

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy



if __name__ == "__main__":

    # params
    alpha = 2
    beta = 0.5
    x0 = 1

    # simulation parameters
    T = 1
    N = 2**7
    dt = 1/N
    t = np.arange(dt,T+dt,dt)
    mc = 100

    E_x_exact = np.zeros((mc,len(t)))

    plt.figure(figsize=(6,4))
    # monte carlo loop
    for i in range(mc):
        np.random.seed(i)
        dW = np.sqrt(dt) * np.random.randn(N)
        W = np.cumsum(dW)
        x_exact = x0 * np.exp(beta*W + (alpha - 0.5*beta**2)*t)
        E_x_exact[i,:] = x_exact
        plt.plot(t,x_exact,'-')

    plt.plot(t,np.mean(E_x_exact,0),'k--',linewidth=3)

    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.grid('on')

    #plt.savefig('plots/exact_reals.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

    plt.show()






