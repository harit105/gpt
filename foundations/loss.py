import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # Small constant to prevent taking log(0) which evaluates to -inf
        epsilon = 1e-7
        
        # Clip probabilities to [epsilon, 1 - epsilon] to avoid log(0) and log(1 - 1)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Formula: -1/n * sum( y * log(p) + (1 - y) * log(1 - p) )
        # np.mean computes the average over all samples (1/n * sum)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        return round(float(loss), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        epsilon = 1e-7
        
        # Prevent log(0) for any class probability
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Formula: -1/n * sum_i( sum_c( y_i,c * log(p_i,c) ) )
        # axis=1 sums over the classes for each individual sample
        # np.mean then averages these sample losses across the batch
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        return round(float(loss), 4)