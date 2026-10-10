import numpy as np

def finite_difference_derivative(coefficients: list, x: float, h: float) -> tuple[float, float, float]:
    """
    Returns the value at x, the value at x plus h, and the estimated slope.
    """
    
   
    coeff=np.array(coefficients[::-1],dtype=float)
    y=np.poly1d(coeff)
    der=(y(x)-y(x+h))/(-h)
    t=(float(y(x)),float(y(x+h)),float(der))
    return t
    
    
