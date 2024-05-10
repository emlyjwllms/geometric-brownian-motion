# plotting and dynamics functions for GBM

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm
import scipy

def gbm_exact(alpha,beta,t,x0,N_iterates):
    N = len(t)
    dt = 1/N
    x_exacts = np.zeros((N_iterates,len(t)))
    # monte carlo loop
    for i in range(N_iterates):
        np.random.seed(i)
        dW = np.sqrt(dt) * np.random.randn(N)
        W = np.cumsum(dW)
        x_exact = x0 * np.exp(beta*W + (alpha - 0.5*beta**2)*t)
        x_exacts[i,:] = x_exact

    return np.mean(x_exacts,0)

def sample_constant_1D(alpha,beta,t,x0,N_iterates,diffusivity_type):
    N = len(t)
    dt = 1/N
    X = np.zeros((N,N_iterates))
    drift = np.zeros((N,N_iterates))
    diffusivity = np.zeros((N,N_iterates))
    X[0,:] = x0

    for m in range(0,N_iterates):
        for n in range(0,N-1):
            dW = np.sqrt(dt) * np.random.randn(1)
            drift[n,m] = alpha
            diffusivity[n,m] = beta
            X[n+1,m] = X[n,m] + drift[n,m]*dt + diffusivity[n,m]*dW
    
        drift[-1,m] = alpha
        diffusivity[-1,m] = beta

    return X.T, drift.T, diffusivity.T

def pascal():
    return np.array([[1,1,1],[1,2,3],[1,3,6]])

def sample_constant_3D(alpha,beta,t,x0,N_iterates,diffusivity_type):
    N = len(t)
    dt = 1/N
    X = np.zeros((3,N,N_iterates))
    drift = np.zeros((3,N,N_iterates))

    if diffusivity_type == 'diagonal':
        diffusivity = np.zeros((3,N,N_iterates))
    if diffusivity_type == 'spd':
        diffusivity = np.zeros((3,3,N,N_iterates))

    X[:,0,:] = np.array([x0[0]*np.ones(N_iterates),x0[1]*np.ones(N_iterates),x0[2]*np.ones(N_iterates)])

    for m in range(0,N_iterates):
        for n in range(0,N-1):
            dW = np.sqrt(dt) * np.random.randn(3)
            drift[:,n,m] = alpha*np.ones_like(x0)
            if diffusivity_type == 'diagonal':
                diffusivity[:,n,m] = (beta*np.ones_like(x0))
                X[:,n+1,m] = X[:,n,m] + drift[:,n,m]*dt + np.matmul(np.diag(diffusivity[:,n,m]),dW)
            if diffusivity_type == 'spd':
                diffusivity[:,:,n,m] = pascal()*beta
                tf = scipy.linalg.issymmetric(diffusivity[:,:,n,m])
                pd = np.all(np.linalg.eigvals(diffusivity[:,:,n,m]) > 0)
                if not (tf and pd):
                    print('Matrix is not SPD')
                    break
                X[:,n+1,m] = X[:,n,m] + drift[:,n,m]*dt + np.matmul(diffusivity[:,:,n,m],dW)

        drift[:,-1,m] = alpha*np.ones_like(x0)
        if diffusivity_type == 'diagonal':
            diffusivity[:,-1,m] = beta*np.ones_like(x0)
        if diffusivity_type == 'spd':
            diffusivity[:,:,-1,m] = pascal()*beta

    return X.T, drift.T, diffusivity.T


def sample_linear_1D(alpha,beta,t,x0,N_iterates,diffusivity_type):
    N = len(t)
    dt = 1/N
    X = np.zeros((N,N_iterates))
    drift = np.zeros((N,N_iterates))
    diffusivity = np.zeros((N,N_iterates))
    X[0,:] = x0

    for m in range(0,N_iterates):
        for n in range(0,N-1):
            dW = np.sqrt(dt) * np.random.randn(1)
            drift[n,m] = alpha*X[n,m]
            diffusivity[n,m] = beta*X[n,m]
            X[n+1,m] = X[n,m] + drift[n,m]*dt + diffusivity[n,m]*dW
    
        drift[-1,m] = alpha*X[-1,m]
        diffusivity[-1,m] = beta*X[-1,m]

    return X.T, drift.T, diffusivity.T

def sample_linear_3D(alpha,beta,t,x0,N_iterates,diffusivity_type):
    N = len(t)
    dt = 1/N
    X = np.zeros((3,N,N_iterates))
    drift = np.zeros((3,N,N_iterates))
    diffusivity = np.zeros((3,N,N_iterates))
    X[:,0,:] = np.array([x0[0]*np.ones(N_iterates),x0[1]*np.ones(N_iterates),x0[2]*np.ones(N_iterates)])

    for m in range(0,N_iterates):
        for n in range(0,N-1):
            dW = np.sqrt(dt) * np.random.randn(3)
            drift[:,n,m] = alpha*X[:,n,m]
            diffusivity[:,n,m] = beta*X[:,n,m]
            X[:,n+1,m] = X[:,n,m] + drift[:,n,m]*dt + np.matmul(np.diag(diffusivity[:,n,m]),dW)
    
        drift[:,-1,m] = alpha*X[:,-1,m]
        diffusivity[:,-1,m] = beta*X[:,-1,m]

    return X.T, drift.T, diffusivity.T

def plot_2(t,data_dictionary,tlabel,xlabel,ylabel,figname,tight=True,show=True,save=True):
    plt.figure(figsize=(8,4))

    for label in data_dictionary:
        data = data_dictionary[label]
        plt.subplot(1,2,1)
        plt.plot(t,data[:,0],label=label)
        plt.xlabel(tlabel)
        plt.ylabel(xlabel)
        plt.ylim(-1,1)
        plt.legend()
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(t,data[:,1],label=label)
        plt.xlabel(tlabel)
        plt.ylabel(ylabel)
        plt.ylim(0,2)
        plt.legend()
        plt.grid(True)

    if tight:
        plt.tight_layout()
    if save:
        plt.savefig(figname,dpi=300,format='png',transparent=True)
    if show:
        plt.show()

    return

def plot_3(t,data_dictionary,tlabel,xlabel,ylabel,zlabel,figname,tight=True,show=True,save=True):
    plt.figure(figsize=(12,4))

    for label in data_dictionary:
        data = data_dictionary[label]
        plt.subplot(1,3,1)
        plt.plot(t,data[:,0],label=label)
        plt.xlabel(tlabel)
        plt.ylabel(xlabel)
        plt.ylim(0,2)
        plt.legend()
        plt.grid(True)

        plt.subplot(1,3,2)
        plt.plot(t,data[:,1],label=label)
        plt.xlabel(tlabel)
        plt.ylabel(ylabel)
        plt.ylim(-1,1)
        plt.legend()
        plt.grid(True)

        plt.subplot(1,3,3)
        plt.plot(t,data[:,2],label=label)
        plt.xlabel(tlabel)
        plt.ylabel(zlabel)
        plt.ylim(0,1)
        plt.legend()
        plt.grid(True)

    if tight:
        plt.tight_layout()
    if save:
        plt.savefig(figname,dpi=300,format='png',transparent=True)
    if show:
        plt.show()

    return