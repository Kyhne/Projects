import numpy as np

def quantize(n, x, xmin, xmax):
    # Handle the special cases where x < xmin or x > xmax
    if x < xmin:
        return '0'*n
    elif x > xmax:
        return '1'*n # jeg blev nødt til at placere dem her, da min while loop ellers er out of bounds
    
    # Generate a linearly spaced sequence X of numbers from xmin to xmax
    X = np.linspace(xmin,xmax,2**n)
    # Generate an empty list of strings
    bitsArray = np.empty(2**n, dtype='object')

    # Fill bitsArray with binary number strings from 0 to 2^(n-1)
    for i in range(2**(n)-1):
        bitsArray[i] = bin(i)
    # Find the smallest k for which X[k] <= x 
    k = 0
    while X[k] <= x:
        k += 1
 
    return bitsArray[k] # mine arrays er forskudt +1 ift. skelettet
	 
def main():
    print('Quantization')
    
    b = quantize(4, 7, 0, 16)
    print(f'7 quantized is {b}')   # result: 111

    b = quantize(2, 2.5, 0, 8)
    print(f'2.5 quantized is {b}') # result: 1

    b = quantize(4, 7, 0, 10)
    print(f'7 quantized is {b}')   # result: 1011

    b = quantize(6, 7, 4.5, 10)
    print(f'7 quantized is {b}')   # result: 11101


if __name__ == "__main__":
    main()
