import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    w=np.zeros(X.shape[1])
    b=0

    for i in range(steps):
        p=_sigmoid(np.dot(X,w)+b)
        l=np.sum(y*np.log(p)+(1-y)*np.log(1-p))
        l=-l/y.shape[0]
        w=w-lr*(np.dot((X.T),(p-y)))/y.shape[0]
        b=b-lr*np.sum((p-y))/y.shape[0]
    
    t=(w,b)
    return t
    
    