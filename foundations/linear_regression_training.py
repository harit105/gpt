import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(
        self, 
        model_prediction: NDArray[np.float64], 
        ground_truth: NDArray[np.float64], 
        N: int, 
        X: NDArray[np.float64], 
        desired_weight: int
    ) -> float:
        # Calculates the partial derivative of MSE loss with respect to weight 'desired_weight'
        # Formula: dL/dw_j = (-2 / N) * sum((Y - Y_hat) * X[:, j])
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(
        self, 
        X: NDArray[np.float64], 
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Forward pass: calculates Y_hat = X * W
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Total number of training samples
        n_samples = len(X)
        
        # Run gradient descent for the specified number of iterations
        for _ in range(num_iterations):
            # Step 1: Forward pass - compute predictions using current weights
            prediction = self.get_model_prediction(X, initial_weights)
            
            # Step 2 & 3: Compute gradients and update each weight
            for j in range(len(initial_weights)):
                # Calculate the gradient of the loss with respect to weight j
                gradient = self.get_derivative(prediction, Y, n_samples, X, j)
                
                # Gradient descent update step: W = W - (alpha * gradient)
                initial_weights[j] -= gradient * self.learning_rate

        # Return the final weights rounded to 5 decimal places as required
        return np.round(initial_weights, 5)