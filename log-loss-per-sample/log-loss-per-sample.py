import math

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of per-sample log loss values.
    """
    L = []
    for i in range(len(y_true)):
        # Clip predictions strictly into [eps, 1 - eps]
        p = min(max(y_pred[i], eps), 1 - eps)
        
        loss = -(y_true[i] * math.log(p) + (1 - y_true[i]) * math.log(1 - p))
        L.append(loss)
    return L
