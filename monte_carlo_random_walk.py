# Hvad er sandsynligheden for at gå til venstre givet r? Det er self 1-r

# Simuler en random walk med np antal partikler og ns antal skridt hvor sandsynligheden for at gå til højre er r. 
# Hint: Lav en variabel x uniformt fordelt mellem  0 og 1 og bestem ud fra denne, om en partikel går til højre eller venstre.
# Det kan vises at den forventede position efter np skridt er ns(2r−1). 
# Sammenlign resultatet i simuleringen med den forventede position ved forskellige værdier for r og ns
import random
import numpy

np = 1000 # Antal partikler
ns = 100  # Antal skridt

def walk_1d(r): # Simulerer en retning, venstre (r-1 sandsynlighed) eller højre (r sandsynlighed [0,1]).
    positioner = numpy.zeros(np) # 1000 nuller der kan blive til venstre eller højre
    for skridt in range(ns):
        for p in range(np): # p = sandsynlighed
            retning = random.uniform(0,1) # Genererer et tilfældigt tal/retning
            if retning <= r:
                positioner[p] += 1 # Går til højre
            else:
                positioner[p] -= 1 # Går til venstre
    return positioner

# Vi laver en funktion, der beregner den forventede position efter np skridt (delopgave 3)

def forventet(r):
    return ns*(2*r-1)

v�rdier = numpy.linspace(0,1, 51) # Genererer forskellige sandsynligheder r mellem 0 og 1, [0, 0.01, 0.02,...]
print(f'r, gennemsnitlig position, forventet position')
for r in v�rdier:
    gennemsnitlig_position = numpy.sum(walk_1d(r))/float(np)
    forventet_position = forventet(r)
    print((r, gennemsnitlig_position, forventet_position)) # Printer ssh. r, den gennemsnitlige(udregnede) position, og den teoretiske position
# Ændrer man på r_værdiers antal af sandsynligheder kan man forkorte og forlænge listen