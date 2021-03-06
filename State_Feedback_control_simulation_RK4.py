import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.pyplot import figure

# Define constants
g = 9.82
l = 0.281
mc = 6.28
alpha = 0.4
mp = 0.175
t_start = 0.
t_end = 30.
km = 0.0934
N = 2000 # Number of steps
x0 = np.array([0.14765, 0., 0., 0.]) # Initial values

# Define A and B and the poles we want
A = np.array([[0., 1., 0., 0.], [(mc+mp)*g/(l*mc), 0., 0., (-alpha)/(l*mc)], [0., 0., 0., 1.], [(g*mp)/mc, 0., 0., (-alpha)/mc]])
B = np.array([[0.], [1./(l*mc)], [0.], [1./mc]])
Poles = np.array([-110, -0.2, complex(-3, 6), complex(-3, -6)])

# Determine K
signal = signal.place_poles(A, B, Poles)
K = signal.gain_matrix

print(f'The K matrix is: {K}')

# Define the model
def model(t,x):
    '''
    Function defining the nonlinear model of the inverted pendulum on a cart,
    with linear state control feedback applied.
    '''
    x1, x2, x3, x4 = x
    u = -np.matmul(K.flatten(),x)
    dx1dt = x2
    dx2dt = (np.cos(x1)*(u-alpha*x4-mp*l*x2**2*np.sin(x1))+(mc+mp)*g*np.sin(x1))/(l*(mc+mp*(1-np.cos(x1)**2)))
    dx3dt = x4
    dx4dt = (u-alpha*x4-mp*l*x2**2*np.sin(x1)+mp*g*np.sin(x1)*np.cos(x1))/(mc+mp*(1-np.cos(x1)**2))
    return np.array([dx1dt, dx2dt, dx3dt, dx4dt])


#Define the RK4 method
def RK4(fcn, a, b, y0, N):
    '''
    Implementation of the RK4 method. Outputs x, which is each step.
    Outputs y, which is the approximation of the solution for each step.

    '''
    h = (b-a)/N
    x = a + np.arange(N+1)*h
    y = np.zeros((x.size,y0.size))
    y[0] = y0
    for k in range(N):
        k1 = fcn(x[k], y[k])
        k2 = fcn(x[k] + h/2, y[k] + h*k1/2)
        k3 = fcn(x[k] + h/2, y[k] + h*k2/2)
        k4 = fcn(x[k] + h, y[k] + h*k3)
        y[k+1] = y[k] + h/6*(k1 + 2*(k2 + k3) + k4)
    return x,y

# Apply the RK4 method to solve the system
t,sol = RK4(model, t_start, t_end, x0, N)
index = np.argmax(abs(sol[:,2])) # Max displacement from the origin
print(f'\nThe biggest deviation from the origin is: {abs(sol[index,2])} meters.')


def plot_angle():
    figure(figsize=(8, 6), dpi=100)
    vinkel = plt.plot(t,sol[:,0], color='b', label=r'Angle $(\theta)$')
    vinkelhst = plt.plot(t,sol[:,1], color='r', label=r'Angular velocity $(\omega)$')
    plt.rc('legend', fontsize=15) 
    plt.rc('ytick', labelsize=15)   
    plt.rc('xtick', labelsize=15)
    plt.legend((vinkel, vinkelhst))
    plt.xlabel('time [s]')
    plt.ylabel(r'$\theta$ [rad] and $\omega$ [rad/s]')
    plt.legend(loc='best')
    plt.rc('axes', labelsize=15)
    plt.grid()
    plt.show()

def plot_placement():
    figure(figsize=(8, 6), dpi=100)
    placering = plt.plot(t,sol[:,2], color='b', label=r'Displacement $(x_c)$')
    hastighed = plt.plot(t,sol[:,3], color='r', label=r'Velocity $(v_c)$')
    plt.rc('legend', fontsize=15) 
    plt.rc('ytick', labelsize=15)   
    plt.rc('xtick', labelsize=15)
    plt.legend((placering, hastighed))
    plt.xlabel('time [s]')
    plt.ylabel(r'$x_c$ [m] and $v$ [m/s]')
    plt.legend(loc='best')
    plt.rc('axes', labelsize=15)
    plt.grid()
    plt.show()

plot_angle() # Plot angle and angular velocity of the cart
plot_placement() # Plot placement and velocity of the cart
