import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        prediction = np.matmul(X, weights)
        #Linear regression computes predictions as a matrix-vector product y^=X.w, which is the same operation as a single neural network layer without an activation function.
        return np.round(prediction,5)
        # Round to 5 decimal places
        pass

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        error = np.mean(np.square(model_prediction - ground_truth))
        #MSE penalizes errors quadratically, making the model focus on reducing its largest mistakes first.

        return round(error,5)
        # Round to 5 decimal places
        pass
"""
    Example Walkthrough:
    -------------------
    Given:                                      [1,2]
      X = [[1, 2],                              [3,4] matrix multiplication(dot product)[0.5,-0.5]
           [3, 4],                              [5,6]
           [5, 6]]                               
      w = [0.5, -0.5]
      y = [0, 1, 0]

    Step              Sample 1               Sample 2               Sample 3
    ---------------------------------------------------------------------------------
    y_pred = X · w     1(0.5) + 2(-0.5) = -0.5  3(0.5) + 4(-0.5) = -0.5  5(0.5) + 6(-0.5) = -0.5
    Error (y_pred - y) -0.5 - 0 = -0.5        -0.5 - 1 = -1.5        -0.5 - 0 = -0.5
    Squared Error     (-0.5)^2 = 0.25        (-1.5)^2 = 2.25        (-0.5)^2 = 0.25

    MSE = (0.25 + 2.25 + 0.25) / 3 = 2.75 / 3 = 0.91667

    Complexity Analysis:
    -------------------
    Time Complexity:
      - get_model_prediction: O(N * d) where N is samples, d is features
      - get_error:            O(N) to compute differences and average
    Space Complexity:
      - O(N) to store the prediction vector
    """