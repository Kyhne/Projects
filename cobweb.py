# cobweb.py:
#   Plot the Logistic Map's quadratic function
#   And show iterating from an initial condition
# Modified version of script from 
# http://csc.ucdavis.edu/~chaos/courses/nlp/

# Import modules
import numpy as np
# Plotting
import matplotlib.pyplot as plt
from pprint import *
# Define the Logistic map's function 
def LogisticMap(r,x):
    return r * x * (1.0 - x)

# Setup x array
# Make sure we have the endpoint x = 1.0
x = np.linspace(0.0, 1.0, 100, endpoint=True)
r = 0.0
# We set the value of the parameter
while r < 3.0:
    r += 0.1
    # We set the initial value
    x0 = 0.1
    
    # Setup the plot
    # It's numbered 1 and is 6 x 6 inches, to make the plot square.
    plt.figure(1,(6,6))
    
    # Note how we turn the parameter value into a string
    #   using the string formating commands.
    TitleString = 'Logistic map: f(x) = %g x (1 - x)' % r
    plt.title(TitleString)
    plt.xlabel('X(n)')   # set x-axis label
    plt.ylabel('X(n+1)') # set y-axis label
    
    # We plot the identity y = x for reference
    plt.plot(x, x, 'b--', antialiased=True)
    
    # Here's the Logistic Map itself
    plt.plot(x, LogisticMap(r,x), 'g', antialiased=True)
    
    # ... and its second iterate
    # Set the initial condition
    state = x0
    
    # Establish the arrays to hold the line end points
    x0 = [ ] # The x_n value
    x1 = [ ] # The next value x_n+1
    # Iterate for a few time steps
    nIterates = 10
    # Plot lines, showing how the iteration is reflected off of the identity
    for n in range(nIterates):
        x0.append( state )
        x1.append( state )
        x0.append( state )
        state = LogisticMap(r,state)
        x1.append( state )
    for i in range(len(x1)):
        if x1[i] != x1[i-1]:# ikke print samme værdi 2 gange
           pprint(x1[i])
    plt.rcParams['figure.dpi'] = 300 # dpi for speed/quality
    # Plot the lines all at once
    plt.plot(x0, x1, 'r', antialiased=True)
    # Display plot in window
    #plt.savefig(f'billede{b}.png') virker ikke lige nu
    plt.show()
