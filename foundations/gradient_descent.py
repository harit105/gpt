class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Initialize x to the starting point
        minimizer = init

        # Perform gradient descent for the specified number of iterations
        for _ in range(iterations):
            # Calculate the derivative of f(x) = x^2
            # f'(x) = 2x
            derivative = 2 * minimizer
            
            # Apply the gradient descent update rule:
            # x_new = x_old - learning_rate * f'(x_old)
            # This moves x one step closer to the minimum (x=0)
            minimizer = minimizer - learning_rate * derivative

        # Round to 5 decimal places and return the final value
        return round(minimizer, 5)