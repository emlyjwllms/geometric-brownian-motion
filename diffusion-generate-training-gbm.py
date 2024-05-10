import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy

# params
alpha = 1
beta = 0.5
x0 = np.array([1,1,1])

# simulation parameters
T = 1
N = 2**9
dt = 1/N
t = np.arange(dt,T+dt,dt)

K = 1000

def pascal():
    return np.array([[1,1,1],[1,2,3],[1,3,6]])

print('loop through elements')
for n in range(0,N-1):
    x_n_current = np.zeros((K,3))
    x_np1_current = np.zeros((K,3))
    for k in range(0,K):
        dW = np.sqrt(dt) * np.random.randn(3)
        x_n_k = x0
        Sigma = pascal()*beta
        tf = scipy.linalg.issymmetric(Sigma)
        pd = np.all(np.linalg.eigvals(Sigma) > 0)
        if not (tf and pd):
            print('Matrix is not SPD')
            break
        x_np1_k = x_n_k + alpha*np.ones_like(x_n_k)*dt + np.matmul(Sigma,dW)
        x_n_current[k,:] = x_n_k
        x_np1_current[k,:] = x_np1_k
    x0 = x_np1_k
    
    if n == 0:
        
        x_n = x_n_current
        x_np1 = x_np1_current
        
    else:
        
        x_n = np.vstack((x_n,x_n_current))
        x_np1 = np.vstack((x_np1,x_np1_current))
    
        
np.savez('data/diffusion-training-data-gbm-3D-spd-constant', x_n=x_n, x_np1=x_np1, dt=dt)