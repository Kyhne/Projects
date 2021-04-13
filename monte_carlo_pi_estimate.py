import random
import numpy as np

N = 1000000
punkter_indeni = 0

for i in range(N):
    x, y = random.uniform(-1,1), random.uniform(-1,1)
    inde_i_cirklen = x**2 + y**2 < 1
    punkter_indeni += inde_i_cirklen

raten_i_cirkel = punkter_indeni / float(N)
rektangel_areal = 4 # fra -1 til 1 på x og y aksen, dvs. 2*2=4
pi_approksimation = float(raten_i_cirkel)*rektangel_areal
fejlen = abs(np.pi - pi_approksimation) # abs -> absolute value/numerisk værdi

print("Pi er approksimeret til: ", pi_approksimation)

print("Fejlen er: ", fejlen)
