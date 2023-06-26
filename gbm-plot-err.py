import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy

gbm_err = np.load('gbm_err.npz')
dt_grid = gbm_err['dt_grid']
str_err_em = gbm_err['str_err_em']
str_err_mil = gbm_err['str_err_mil']
str_err_rk = gbm_err['str_err_rk']
str_err_tay = gbm_err['str_err_tay']
weak_err_em = gbm_err['weak_err_em']
weak_err_mil = gbm_err['weak_err_mil']
weak_err_rk = gbm_err['weak_err_rk']
weak_err_tay = gbm_err['weak_err_tay']

plt.figure()
h1, = plt.loglog(dt_grid,str_err_em,'.-',color='tab:blue',label="Euler-Maruyama")
h2, = plt.loglog(dt_grid,str_err_mil,'.-',color='tab:orange',label="Milstein")
h3, = plt.loglog(dt_grid,str_err_rk,'.-',color='tab:green',label="Runge-Kutta")
h4, = plt.loglog(dt_grid,str_err_tay,'.-',color='tab:red',label="Taylor")
h5, = plt.loglog(dt_grid,weak_err_em,'.--',color='tab:blue')
h6, = plt.loglog(dt_grid,weak_err_mil,'.--',color='tab:orange')
h7, = plt.loglog(dt_grid,weak_err_rk,'.--',color='tab:green')
h8, = plt.loglog(dt_grid,weak_err_tay,'.--',color='tab:red')

l1, = plt.loglog(0,0,'k-',label="Strong")
l2, = plt.loglog(0,0,'k--',label="Weak")

plt.loglog(dt_grid,10*dt_grid**0.5,'-.',color='grey')
plt.text(2*10**(-3),1,r'$\mathcal{O}(\Delta t^{0.5})$')
plt.loglog(dt_grid,7*dt_grid,'-.',color='grey')
plt.text(10**(-3),3*10**(-3),r'$\mathcal{O}(\Delta t)$')
plt.loglog(dt_grid,5*dt_grid**1.5,'-.',color='grey')
plt.text(10**(-3),8*10**(-5),r'$\mathcal{O}(\Delta t^{1.5})$')
plt.loglog(dt_grid,5*dt_grid**2,'-.',color='grey')
plt.text(2*10**(-3),10**(-5),r'$\mathcal{O}(\Delta t^2)$')

first_legend = plt.legend(title="Method",handles=[h1,h2,h3,h4,h5,h6,h7,h8], loc='lower right', fontsize='8',title_fontsize='8')
ax = plt.gca().add_artist(first_legend)
plt.legend(title="Error",handles=[l1,l2], loc='upper left',fontsize='8',title_fontsize='8')
plt.grid(which="both",alpha=0.25)
plt.ylim(10**(-6),5)
plt.xlabel(r'$\Delta t$')
plt.ylabel(r'$e$')

plt.savefig('gbm-err.png',dpi=300,format='png')

plt.show()