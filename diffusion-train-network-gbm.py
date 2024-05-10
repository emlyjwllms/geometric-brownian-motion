import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy

from gbm_functions import *

import tensorflow as tf
from Network_DietrichBased import (
                                    SDEIdentification,
                                    ModelBuilder,
                                    SDEApproximationNetwork
                                  )

training_data = np.load('data/diffusion-training-data-gbm-3D-diagonal-linear-sigma.npz')
xn = training_data['x_n']
xnp1 = training_data['x_np1']
dt = float(training_data['dt'])

train = False

n_layers = 1 #Number of hidden layers
n_dim_per_layer = 2 #Neurons per layer

n_dimensions = 3 #Spatial dimension 

ACTIVATIONS = tf.nn.leaky_relu #Activation function
VALIDATION_SPLIT = .2 # 80% for training, 20% for testing
BATCH_SIZE = 32
LEARNING_RATE = 1e-1
N_EPOCHS = 50

# sigma matrix 
diffusivity_type = "diagonal"

tf.random.set_seed(1)

encoder = ModelBuilder.define_gaussian_process(
                                    n_input_dimensions=n_dimensions,
                                    n_output_dimensions=n_dimensions,
                                    n_layers=n_layers,
                                    n_dim_per_layer=n_dim_per_layer,
                                    name="diff_net",
                                    activation=ACTIVATIONS,
                                    diffusivity_type=diffusivity_type)
# encoder.summary()

file_path = 'Trained_Dietrich_GBM_3D_linear_sigma'
file_path += '/' + diffusivity_type + '/'
file_path += f'HL{n_layers}_'
file_path += f'N{n_dim_per_layer}_'
file_path += 'RELU_'
file_path += 'LR1e-1_'
file_path += f'BS{BATCH_SIZE}_'
file_path += f'EP{N_EPOCHS}/'

if train:

    n_pts = xnp1.shape[0]

    step_sizes = np.zeros(n_pts) + dt

    model = SDEApproximationNetwork(sde_model=encoder,
                                    method="euler",
                                    diffusivity_type=diffusivity_type)

    model.compile(optimizer=tf.keras.optimizers.Adamax())

    sde_i = SDEIdentification(model=model)

    hist = sde_i.train_model(xn, xnp1, step_size=step_sizes,
                            validation_split=VALIDATION_SPLIT,
                            n_epochs=N_EPOCHS,
                            batch_size=BATCH_SIZE)

    plt.figure(16,figsize=(6,4))
    # plt.title(r"$\Sigma$")
    plt.plot(hist.history["loss"], label='Training')
    plt.plot(hist.history["val_loss"], label='Validation')
    plt.ylim([np.min(hist.history["loss"])*1.1, np.max(hist.history["loss"])])
    #plt.ylim([-5,50])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(file_path + 'plots/training.png')
    plt.show()

    model.save_weights(file_path, overwrite=True, save_format=None, options=None)

model = SDEApproximationNetwork(sde_model=encoder,
                                step_size=dt,
                                method="euler",
                                diffusivity_type=diffusivity_type)

model.load_weights(file_path).expect_partial()

sde_i = SDEIdentification(model=model)

# params
alpha = 0
beta = 0.5
x0 = np.array([1,1,1])

# simulation parameters
T = 1
N = 2**9
dt = 1/N
t = np.arange(dt,T+dt,dt)

N_iterates = 10000

X, drift, diffusivity = sample_linear_3D(alpha,beta,t,x0,N_iterates,diffusivity_type)
print(np.shape(X))
print(np.shape(drift))
print(np.shape(diffusivity))


X_NN = sde_i.sample_path(x0,dt,N-1,N_iterates)
print(np.shape(X_NN))

drift_NN = np.zeros_like(X_NN)
diffusivity_NN = np.zeros_like(X_NN)
for n in range(0,N_iterates):
    drift_NN[n,:], diffusivity_NN[n,:] = sde_i.drift_diffusivity(np.squeeze(X_NN)[n,:])

E_X = np.mean(X,0)
print(np.shape(E_X))
E_X_NN = np.mean(np.squeeze(X_NN),0)
print(np.shape(E_X_NN))

# linear GBM exact solution
# E_X_exact = gbm_exact(alpha,beta,t,x0,N_iterates)

E_drift = np.mean(drift,0)
E_diffusivity = np.mean(diffusivity,0)

E_drift_NN = np.mean(drift_NN,0)
E_diffusivity_NN = np.mean(diffusivity_NN,0)

# deterministic relationships
xmin = min(np.min(xn),np.min(xnp1))
xmax = max(np.max(xn),np.max(xnp1))
X_range = np.vstack([np.linspace(xmin,xmax,100).T,np.linspace(xmin,xmax,100).T,np.linspace(xmin,xmax,100).T]).T

print(np.shape(X_range))

drift_NN_deterministic, diffusivity_NN_deterministic = sde_i.drift_diffusivity(X_range)
drift_deterministic = alpha*np.zeros_like(X_range)
diffusivity_deterministic = beta*X_range

print("saving data")
np.savez('data/paths-NN-GBM-3D-diagonal-linear-sigma.npz', t=t, X_range=X_range, drift_deterministic=drift_deterministic, diffusivity_deterministic=diffusivity_deterministic, drift_NN_deterministic=drift_NN_deterministic, diffusivity_NN_deterministic=diffusivity_NN_deterministic, E_X=E_X, E_X_NN=E_X_NN, E_drift=E_drift, E_diffusivity=E_diffusivity, E_drift_NN=E_drift_NN, E_diffusivity_NN=E_diffusivity_NN)

# Sigma_x_NN = np.zeros(N)
# Sigma_y_NN = np.zeros(N)
# Sigma_z_NN = np.zeros(N)
# for n in range(0,N):
#     Sigma_x_NN[n] = np.matmul(std[n,0,:],X[n,:])
#     Sigma_y_NN[n] = np.matmul(std[n,1,:],X[n,:])
#     Sigma_z_NN[n] = np.matmul(std[n,2,:],X[n,:])


