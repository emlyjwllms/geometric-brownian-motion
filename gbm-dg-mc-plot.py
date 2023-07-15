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


gbm_dg_mc = np.load('gbm_dg_mc.npz')
t = gbm_dg_mc['t']
E_x_exact = gbm_dg_mc['E_x_exact']
E_xhs = gbm_dg_mc['E_xhs']
mcs = gbm_dg_mc['mcs']


plt.figure(figsize=(4,6))
plt.plot(t,E_x_exact,'k-',linewidth=2,zorder=3,label="Analytical")
plt.plot(t,E_x_exact,color='tab:blue',linewidth=2,zorder=2,label="DG")

for mci in range(len(mcs)):
    E_xh = E_xhs[mci]
    plt.plot(t,E_xh,zorder=1,color="tab:blue",linewidth=2, alpha=mci/len(mcs))

plt.xlabel(r"$t$")
plt.ylabel(r"$x$")
plt.grid()
#plt.legend()

plt.savefig('plots/gbm-dg-mc.png',dpi=300,format='png',transparent=True,bbox_inches='tight')
plt.show()