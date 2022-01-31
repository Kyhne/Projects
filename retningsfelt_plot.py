import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

def model(x,t):
    y = x
    dxdt = y*(y-2)
    return dxdt
xrange = 3  
t = np.linspace(0,xrange,1001)

x0 = [1.8]
fig, ax = plt.subplots(figsize=(10,6))
plt.xlabel('$t$', fontsize=15, x=1.0)
plt.ylabel(r'$y$', y=1., fontsize = 16,rotation=0)
y = odeint(model,x0,t)

# Placer x -aksen ved x=0
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.plot(t,y[:,0],'b', label = r'$y$')
plt.rc('legend', fontsize=13) 
X, Y = np.meshgrid(np.arange(0, xrange, .25), np.arange(-3, 3, .30))
U = np.ones_like(X) #dydt = 1
V = Y*(Y-2)
magnitude = np.sqrt(U**2 + V**2)
U = U/magnitude
V = V/magnitude
ax.quiver(X, Y, U, V, angles = 'xy')
plt.title(r'$\frac{\mathrm{d}y}{\mathrm{d}t} = y(y-2)$',fontsize = 16)
plt.ylim(-2,2)
plt.xlim(0,3)
plt.legend(loc = 'best')
plt.figure(figsize=(12,6))
plt.rcParams['figure.dpi'] = 150
plt.show()