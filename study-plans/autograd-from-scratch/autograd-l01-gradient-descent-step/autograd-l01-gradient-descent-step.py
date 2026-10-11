import numpy as np

def gradient_descent_step(values: list, gradients: list, learning_rate: float) -> tuple[list, float]:
    """
    Returns a fresh list of updated values and the predicted objective change.
    """
    values=np.array(values,dtype=float)
    grad=np.array(gradients,dtype=float)
    values=values-learning_rate*grad
    l=-learning_rate*np.dot(grad,grad)
    values=values.tolist()
    t=(values,float(l))
    return t