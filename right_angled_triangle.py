"""
    Docstring om modulet.
    Denne fil indeholder en funktion, der bestemmer arealet af en retvinklet trekant.

"""
def f(a,b):
    
    """
    Parameters
    ----------
    a : Højden af den retvinklede trekant.
    b : Bredden af den retvinklede trekant.

    Returns
    -------
    Arealet af den retvinklede trekant

    Examples
    -------
    >>> f(1,2)
    1.0
    >>> f(3,4)
    6.0
    >>> f(10,10)
    50.0
    """
    return 1/2*a*b

if __name__ == '__main__':
    import doctest
    doctest.testmod()

print(__doc__)
print(f.__doc__)