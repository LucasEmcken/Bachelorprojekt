import numpy as np
from scipy.optimize import minimize

def nnls(A, b, alpha=0, x0=None):
    """Non-negative least squares with L1 regularization.

    Parameters
    ----------
    A : ndarray
        The input data matrix.
    b : ndarray
        The target values.
    alpha : float
        The regularization parameter.
    x0 : ndarray, optional
        The initial guess for the solution. If None, a zero vector is used.

    Returns
    -------
    x : ndarray
        The solution to the problem.
    """
    def objective_function(x, A, b, alpha):
        """Objective function for L1 regularized NNLS"""
        return np.linalg.norm(A @ x - b)**2 + alpha * np.sum(np.abs(x))

    # Use a zero vector as the initial guess if not provided
    if x0 is None:
        x0 = np.zeros(A.shape[1])

    # Define bounds for non-negativity
    bounds = [(0, None) for _ in range(A.shape[1])]

    # Minimize the objective function
    result = minimize(objective_function, x0, args=(A, b, alpha), method='L-BFGS-B', bounds=bounds)

    return result.x
