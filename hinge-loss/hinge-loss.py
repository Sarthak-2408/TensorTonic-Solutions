import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    y_true=np.array(y_true,dtype=float)
    y_score=np.array(y_score,dtype=float)
    m=np.ones(y_true.shape,dtype=float)
    m=margin*m
    l=np.maximum(0,m-y_true*y_score)
    if reduction =="mean":
        return float(np.mean(l))
    if reduction == "sum":
        return float(np.sum(l))
    