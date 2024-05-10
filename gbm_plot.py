import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy

from gbm_functions import *

gbm =  np.load('data/paths-NN-GBM-3D-diagonal-linear-sigma.npz')
E_X = gbm['E_X']
E_X_NN = gbm['E_X_NN']
# E_X_exact = gbm['E_X_exact']
E_drift = gbm['E_drift']
E_diffusivity = gbm['E_diffusivity']
E_drift_NN = gbm['E_drift_NN']
E_diffusivity_NN = gbm['E_diffusivity_NN']
drift_NN_deterministic = gbm['drift_NN_deterministic']
diffusivity_NN_deterministic = gbm['diffusivity_NN_deterministic']
drift_deterministic = gbm['drift_deterministic']
diffusivity_deterministic = gbm['diffusivity_deterministic']
t = gbm['t']
X_range = gbm['X_range']


# plt.figure(figsize=(4,4))
# plt.plot(t,E_X_exact,label=r"$X_{\mathrm{exact}}$")
# plt.plot(t,E_X,label=r"$X$")
# plt.plot(t,E_X_NN,label=r"$X_{NN}$")
# plt.xlabel(r"t")
# plt.ylabel(r"$E[X]$")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('plots/paths-NN-1D-linear-exact.png',dpi=300,format='png',transparent=True)
# plt.show()

plot_3(t, {r"$X_1$": np.vstack([E_X[:,0].T,E_drift[:,0].T,E_diffusivity[:,0].T]).T, r"$X_1^{NN}$": np.vstack([E_X_NN[:,0].T,E_drift_NN[:,0].T,E_diffusivity_NN[:,0].T]).T}, r"t",r"$E[X_1]$",r"$E[f]$",r"$E[\sigma]$",'plots/paths-NN-3D-diagonal-linear-sigma-time-X.png',save=True)
plot_3(t, {r"$X_2$": np.vstack([E_X[:,1].T,E_drift[:,1].T,E_diffusivity[:,1].T]).T, r"$X_2^{NN}$": np.vstack([E_X_NN[:,1].T,E_drift_NN[:,1].T,E_diffusivity_NN[:,1].T]).T}, r"t",r"$E[X_2]$",r"$E[f]$",r"$E[\sigma]$",'plots/paths-NN-3D-diagonal-linear-sigma-time-Y.png',save=True)
plot_3(t, {r"$X_3$": np.vstack([E_X[:,2].T,E_drift[:,2].T,E_diffusivity[:,2].T]).T, r"$X_3^{NN}$": np.vstack([E_X_NN[:,2].T,E_drift_NN[:,2].T,E_diffusivity_NN[:,2].T]).T}, r"t",r"$E[X_3]$",r"$E[f]$",r"$E[\sigma]$",'plots/paths-NN-3D-diagonal-linear-sigma-time-Z.png',save=True)

# plot_3(t, {r"$X$": np.vstack([E_X.T,E_drift.T,E_diffusivity.T]).T, r"$X_{NN}$": np.vstack([E_X_NN.T,E_drift_NN.T,E_diffusivity_NN.T]).T}, r"t",r"$E[X]$",r"$E[f]$",r"$E[\sigma]$",'plots/paths-NN-1D-linear-sigma-time.png',save=True)
plot_2(X_range[:,0], {r"$\mathrm{analytic}$": np.vstack([drift_deterministic[:,0].T,diffusivity_deterministic[:,0].T]).T, r"$NN$": np.vstack([drift_NN_deterministic[:,0].T,diffusivity_NN_deterministic[:,0].T]).T}, r"X",r"$f$",r"$\sigma$",'plots/paths-NN-3D-diagonal-linear-sigma-space-X.png',save=True)
plot_2(X_range[:,1], {r"$\mathrm{analytic}$": np.vstack([drift_deterministic[:,1].T,diffusivity_deterministic[:,1].T]).T, r"$NN$": np.vstack([drift_NN_deterministic[:,1].T,diffusivity_NN_deterministic[:,1].T]).T}, r"X",r"$f$",r"$\sigma$",'plots/paths-NN-3D-diagonal-linear-sigma-space-Y.png',save=True)
plot_2(X_range[:,2], {r"$\mathrm{analytic}$": np.vstack([drift_deterministic[:,2].T,diffusivity_deterministic[:,2].T]).T, r"$NN$": np.vstack([drift_NN_deterministic[:,2].T,diffusivity_NN_deterministic[:,2].T]).T}, r"X",r"$f$",r"$\sigma$",'plots/paths-NN-3D-diagonal-linear-sigma-space-Z.png',save=True)

if 0:
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(E_X[:,0],E_drift[:,0],label=r"$X_1$")
    plt.plot(E_X_NN[:,0],E_drift_NN[:,0],label=r"$X_1^{NN}$")
    plt.xlabel(r"$E[X_1]$")
    plt.ylabel(r"$E[f]$")
    plt.ylim(-1,1)
    plt.xlim(0,2)
    plt.legend()
    plt.grid(True)
    plt.subplot(1,3,2)
    plt.plot(E_X[:,1],E_drift[:,1],label=r"$X_2$")
    plt.plot(E_X_NN[:,1],E_drift_NN[:,1],label=r"$X_2^{NN}$")
    plt.xlabel(r"$E[X_2]$")
    plt.ylabel(r"$E[f]$")
    plt.ylim(-1,1)
    plt.xlim(0,2)
    plt.legend()
    plt.grid(True)
    plt.subplot(1,3,3)
    plt.plot(E_X[:,2],E_drift[:,2],label=r"$X_3$")
    plt.plot(E_X_NN[:,2],E_drift_NN[:,2],label=r"$X_3^{NN}$")
    plt.xlabel(r"$E[X_3]$")
    plt.ylabel(r"$E[f]$")
    plt.ylim(-1,1)
    plt.xlim(0,2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/paths-NN-3D-diagonal-linear-sigma-space-f.png',dpi=300,format='png',transparent=True)
    plt.show()

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(E_X[:,0],E_diffusivity[:,0],label=r"$X_1$")
    plt.plot(E_X_NN[:,0],E_diffusivity_NN[:,0],label=r"$X_1^{NN}$")
    plt.xlabel(r"$E[X_1]$")
    plt.ylabel(r"$E[\sigma]$")
    plt.ylim(0,1)
    plt.xlim(0,2)
    plt.legend()
    plt.grid(True)
    plt.subplot(1,3,2)
    plt.plot(E_X[:,1],E_diffusivity[:,1],label=r"$X_2$")
    plt.plot(E_X_NN[:,1],E_diffusivity_NN[:,1],label=r"$X_2^{NN}$")
    plt.xlabel(r"$E[X_1]$")
    plt.ylabel(r"$E[\sigma]$")
    plt.ylim(0,1)
    plt.xlim(0,2)
    plt.legend()
    plt.grid(True)
    plt.subplot(1,3,3)
    plt.plot(E_X[:,2],E_diffusivity[:,2],label=r"$X_3$")
    plt.plot(E_X_NN[:,2],E_diffusivity_NN[:,2],label=r"$X_3^{NN}$")
    plt.xlabel(r"$E[X_1]$")
    plt.ylabel(r"$E[\sigma]$")
    plt.ylim(0,1)
    plt.xlim(0,2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/paths-NN-3D-diagonal-linear-sigma-space-sigma.png',dpi=300,format='png',transparent=True)
    plt.show()