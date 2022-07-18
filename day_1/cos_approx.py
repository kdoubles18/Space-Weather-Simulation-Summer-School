#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Kaitlin Doublestein'
__email__ = 'kdoubles@umich.edu'

from math import factorial
from math import pi
import numpy as np
import matplotlib.pyplot as plt

def cos_approx(x, accuracy=10):
    """ Returns the approximation of cosine using the Taylor expansion form.
    Args: 
        x(float):
            To evaluate cosine of.
        accuracy (int):
            (default: 10) Number of Taylor series coefficients to use.
    
    Returns:
        (float):Approximate cosine of *x*.
    
    Examples:
        from math import pi
        cos_approx(pi)
        cos_approx(pi,accuracy=50)
    """
    cos_approximation = sum([ ((-1)**n/factorial(2*n)) * (x**(2*n)) for n in range(accuracy) ])
    return cos_approximation

#Could rewrite so that the coefficients are separate from the series/
#coefficient  = ((-1)**n/factorial(2*n))
#series = (x**(2*n))
#sum([(cofficient*series) for n in range(accuracy)])

def cos_approx_sep(x,accuracy=10):
    """Returns the approximation of cosine using the Taylor expansion form.
    Args: 
        x(float):
            To evaluate cosine of.
        accuracy (int):
            (default: 10) Number of Taylor series coefficients to use.
    
    Returns:
        (float):Approximate cosine of *x*.
    
    Examples:
        from math import pi
        cos_approx(pi)
        cos_approx(pi,accuracy=50)
    """
    def coeff(n)
    return 


# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    print("cos(0) = ", cos_approx(0))
    print("cos(pi) = ", cos_approx(pi))
    print("cos(2*pi) = ", cos_approx(2*pi))
    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
    
    

x = np.linspace(0,1)
plt.plot(x, np.exp(x))
plt.xlabel(r'$0 \leq x < 1$')
plt.ylabel(r'$e^x$')
plt.title(r'Exponential Function')
plt.show()

