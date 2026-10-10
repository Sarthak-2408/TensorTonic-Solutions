import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    # Write code here
    y_pred=np.array(y_pred,dtype=float)
    y_true=np.array(y_true,dtype=float)
    a=(y_pred-y_true)**2
    m=np.sum(a)
    return float(m/y_true.shape[0])