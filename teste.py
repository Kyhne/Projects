import math

def circle_area(r):
    return(math.pi*r**2)
 
def circle_circumference(r):
    return(2*math.pi*r)

if __name__ == '__main__':
    r = 1
    tolerance = 1e-5
    test1 = abs(circle_area(r) - math.pi) < tolerance
    test2 = abs(circle_circumference(r) - 2*math.pi) < tolerance
    if test1 and test2:
        print('All tests have passed')
    else:
        print('At least one test failed!')