import numpy as np
import matplotlib.pyplot as plt
import math
def main():
    x=np.arange(-np.pi,np.pi,0.001) 

    n=1000 #antal led

    An = np.array([]) # et array, der indeholder alle a_n værdier

    a0 = -(4*(math.pi)**2)/6  # definerer a_0

    for i in range(n):
        an=(-8*(-1)**(i+1))/(i+1)**2 # regner vores a_n værdier udfra formlen
        An = np.append(An, an) # Tilføjer værdierne til vores tomme array

    for i in range(n):
        if i==0.0: # vi skal huske at bruge a_0 ved første led/iteration
            sum=a0
        else:
            sum=sum+(An[i-1]*np.cos(i*x)) # Hele Fourierrækken

    plt.plot(x,sum,'-b')
    plt.title("Fourier serie for vores funktion")
    plt.show() 
    
if __name__ == '__main__':
    main()