import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        ## Step 1: Subtract max(z) for numerical stability.
        # This ensures the maximum exponent is e^0 = 1, preventing overflow to infinity.
        shifted = z - np.max(z)
        
        # Step 2: Compute e^(z_i - max(z)) for each element.
        exps = np.exp(shifted)

        # Step 3: Normalize by dividing each element by the sum of all elements.
        # This converts the values into a valid probability distribution (sums to 1).
        # np.round(..., 4) rounds each value to 4 decimal places as required.
        return np.round(exps/np.sum(exps), 4)

        pass
