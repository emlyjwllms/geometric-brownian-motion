# geometric Brownian motion error

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy

def eulermaruyama(x0,alpha,beta,dW,N,dt):

    x_em = []
    x = x0

    for i in range(N):
        x = x + alpha*x*dt + beta*x*dW[i]
        x_em = np.append(x_em,x)

    return x_em

def milstein(x0,alpha,beta,dW,N,dt):

    x_mil = []
    x = x0

    for i in range(N):
        x = x + alpha*x*dt + beta*x*dW[i] + 0.5*beta**2*x*(dW[i]**2 - dt)
        x_mil = np.append(x_mil,x)

    return x_mil

def rk(x0,alpha,beta,dW,N,dt):

    x_rk = []
    x = x0

    for i in range(N):
        x_tilde = x + alpha*x*dt + beta*x*np.sqrt(dt)
        x = x + alpha*x*dt + beta*x*dW[i] + 1/(2*np.sqrt(dt))*(beta*x_tilde - beta*x)*(dW[i]**2 - dt)
        x_rk = np.append(x_rk,x)

    return x_rk

def taylor(x0,alpha,beta,dW,N,dt):

    x_tay = []
    x = x0

    dV = np.sqrt(dt) * np.random.normal(0,np.sqrt(dt),size=N)

    for i in range(N):
        dZ = 0.5*dt*(dW[i] + dV[i]/np.sqrt(3))
        x = x + alpha*x*dt + beta*x*dW[i] + 0.5*beta**2*x*(dW[i]**2 - dt) + alpha*beta*x*dZ + 0.5*alpha**2*x*dt**2 + alpha*beta*x*(dW[i]*dt - dZ) + 0.5*beta**3*x*((1/3)*dW[i]**2-dt)*dW[i]
        x_tay = np.append(x_tay,x)

    return x_tay

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

    mc = 10**5
    
    x_exact = 0
    x_em = 0
    x_mil = 0
    x_rk = 0
    x_tay = 0

    # mc loop
    for i in range(mc):
        np.random.seed(i)
        dW = np.sqrt(dt) * np.random.randn(N)
        W = np.cumsum(dW)

        x_exact += x0 * np.exp(beta*W + (alpha - 0.5*beta**2)*t)
        x_em += eulermaruyama(x0,alpha,beta,dW,N,dt)
        x_mil += milstein(x0,alpha,beta,dW,N,dt)
        x_rk += rk(x0,alpha,beta,dW,N,dt)
        x_tay += taylor(x0,alpha,beta,dW,N,dt)

    plt.figure()
    plt.plot(t,x_exact/mc,'k-',linewidth=2,label="Analytical")
    plt.plot(t,x_em/mc,label="Euler-Maruyama")
    plt.plot(t,x_mil/mc,label="Milstein")
    plt.plot(t,x_rk/mc,label="Runge-Kutta")
    plt.plot(t,x_tay/mc,label="Taylor")

    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.grid('on')
    plt.legend()
    plt.show()

    str_err_em = []
    weak_err_em = []
    str_err_mil = []
    weak_err_mil = []
    str_err_rk = []
    weak_err_rk = []
    str_err_tay = []
    weak_err_tay = []
    dt_grid = []

    for R in np.arange(0,5,2, dtype=float):
        dt_grid = np.append(dt_grid,2**(R-10))

    for Dt_i in range(len(dt_grid)):
        Dt = dt_grid[Dt_i]
        t = np.arange(Dt,1+Dt,Dt)
        n = len(t)

        err_em = 0
        err_mil = 0
        err_rk = 0
        err_tay = 0
        x_exact_sum = 0
        x_em_sum = 0
        x_mil_sum = 0
        x_rk_sum = 0
        x_tay_sum = 0

        for i in range(mc):
            np.random.seed(i)
            #dW = np.sqrt(Dt) * np.random.normal(0,np.sqrt(Dt),size=n)
            dW = np.sqrt(Dt) * np.random.randn(n)
            W = np.cumsum(dW)
            x_exact = x0 * np.exp(beta*W + (alpha - 0.5*beta**2)*t)

            x_em = eulermaruyama(x0,alpha,beta,dW,n,Dt)
            x_mil = milstein(x0,alpha,beta,dW,n,Dt)
            x_rk = rk(x0,alpha,beta,dW,n,Dt)
            x_tay = taylor(x0,alpha,beta,dW,n,Dt)

            err_em += np.abs(x_exact - x_em)
            err_mil += np.abs(x_exact - x_mil)
            err_rk += np.abs(x_exact - x_rk)
            err_tay += np.abs(x_exact - x_tay)

            x_exact_sum += x_exact
            x_em_sum += x_em
            x_mil_sum += x_mil
            x_rk_sum += x_rk
            x_tay_sum += x_tay

        str_err_em = np.append(str_err_em, np.max(err_em/mc))
        str_err_mil = np.append(str_err_mil, np.max(err_mil/mc))
        str_err_rk = np.append(str_err_rk, np.max(err_rk/mc))
        str_err_tay = np.append(str_err_tay, np.max(err_tay/mc))

        weak_err_em = np.append(weak_err_em, np.max(np.abs(x_exact_sum - x_em_sum)/mc))
        weak_err_mil = np.append(weak_err_mil, np.max(np.abs(x_exact_sum - x_mil_sum)/mc))
        weak_err_rk = np.append(weak_err_rk, np.max(np.abs(x_exact_sum - x_rk_sum)/mc))
        weak_err_tay = np.append(weak_err_tay, np.max(np.abs(x_exact_sum - x_tay_sum)/mc))

    np.savez('gbm_err', dt_grid=dt_grid, str_err_em=str_err_em, str_err_mil=str_err_mil, str_err_rk=str_err_rk, str_err_tay=str_err_tay,
             weak_err_em=weak_err_em, weak_err_mil=weak_err_mil, weak_err_rk=weak_err_rk, weak_err_tay=weak_err_tay )
