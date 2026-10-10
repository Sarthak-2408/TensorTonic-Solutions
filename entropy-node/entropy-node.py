import numpy as np
def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    y = np.array(y)
    # Get unique classes and their counts
    values, counts = np.unique(y, return_counts=True)
    probs = counts / y.shape[0]
    
    # Filter out zero probabilities to avoid log2(0)
    H = -np.sum(probs * np.log2(probs))
    return float(H)