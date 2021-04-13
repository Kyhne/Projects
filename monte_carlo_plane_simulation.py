import random
import math
import numpy as np

random.seed(1) # Denne skal slettes, medmindre der testes
antal_fly=200 
# np.random.poisson(200,1) var mere korrekt, men over mange iterationer er gennemsnittet 200 alligevel.

tid_det_tager_at_lande = np.array(random.choices([15, 45, 75, 105, 135, 165, 195, 225, 255, 285], weights=[0, 8, 16.5, 30.5, 20.5, 12.5, 5, 4, 3, 0], k=antal_fly))
fly_ankomst_luftrum=[]
ventetid_mellem_fly=[]
delta_tid=0
delta_tid_array=[]
ventetid_gennemsnit_pr_dag=[]
Gennemsnitlig_ventetid_for_et_fly=0
# Definerer vores monte-carlo simulerings funktion
def monte_carlo(iterations): #Tager parameteren iterations som er gentagelser
    """
    Udregner den gennemsnitlige ventetid for fly i luften inden de lander
    
    Input:
    gentageler - antal gange vi skal køre simuleringen
    
    Returns:
    Den gennemsnitlige ventetid for hvert enkelt fly.
    Examples:
    >>> monte_carlo(10)
    28.20842935505075
    """
    for n in range(iterations): 
        for i in range(antal_fly):
            # Anvender den inverse CDF metode til at bestemme ventetiden mellem fly
            ventetid_mellem_fly.append(-math.log(1.0-random.random())/0.00411522633) # denne værdi er antal_fly/48600 sekunder

			# Vi ønsker summen af alle ventetider for at få den gennemsnitlige ventetid
            x=sum(ventetid_mellem_fly)
			#jeg tager summen af alle tiderne mellem flyene
            fly_ankomst_luftrum.append(x)
			
				
			#lidt matematik
            if i>0 and (tid_det_tager_at_lande[i-1]+fly_ankomst_luftrum[i-1]+delta_tid)>fly_ankomst_luftrum[i]:
                # print(fly_ankomst_luftrum[i],'Ventetid')
                delta_tid=(tid_det_tager_at_lande[i-1]+fly_ankomst_luftrum[i-1])-fly_ankomst_luftrum[i]
                delta_tid_array.append(delta_tid)
            else:      				
                #print(fly_ankomst_luftrum[i],'OK')
                delta_tid=0
                delta_tid_array.append(0)
			#Bestemmer den gennemsnitlige ventetid på en hel dag
            ventetid_gennemsnit_pr_dag.append(np.average(delta_tid_array))	

	#print(ventetid_gennemsnit_pr_dag)
    Gennemsnitlig_ventetid_for_et_fly=np.average(ventetid_gennemsnit_pr_dag)
    return Gennemsnitlig_ventetid_for_et_fly
# Vi kører 100 iterationer, da flere tager meget lang tid, især når antallet af fly øges markant
print(monte_carlo(100))

# Vi tester det hele. Det nedenunder skal slettes medmindre man tester
if __name__ == '__main__':
    import doctest
    doctest.testmod()
    
print(__doc__)