import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy

gbm_err = np.load('gbm_dg_err.npz')
dt_grid = gbm_err['dt_grid']
str_err_dg = gbm_err['str_err_dg']
weak_err_dg = gbm_err['weak_err_dg']


plt.figure()
h1, = plt.loglog(dt_grid,str_err_dg[0,:],'.-',color='tab:blue',label=r"p = 1")
h2, = plt.loglog(dt_grid,weak_err_dg[0,:],'.--',color='tab:blue')

l1, = plt.loglog(0,0,'k-',label="Strong")
l2, = plt.loglog(0,0,'k--',label="Weak")

plt.loglog(dt_grid,10*dt_grid**0.5,'-.',color='grey')
plt.text(2*10**(-3),1,r'$\mathcal{O}(\Delta t^{0.5})$',fontsize='14')
plt.loglog(dt_grid,7*dt_grid,'-.',color='grey')
plt.text(10**(-3),3*10**(-3),r'$\mathcal{O}(\Delta t)$',fontsize='14')

first_legend = plt.legend(handles=[h1,h2], loc='lower right')
ax = plt.gca().add_artist(first_legend)
plt.legend(title="Error",handles=[l1,l2], loc='upper left')
plt.grid(which="both",alpha=0.25)
plt.ylim(10**(-3),5)
plt.xlabel(r'$\Delta t$')
plt.ylabel(r'$e$')

plt.savefig('plots/gbm-dg-err.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

plt.show()