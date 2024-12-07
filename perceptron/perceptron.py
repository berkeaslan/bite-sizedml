import numpy as np

class Perceptron:
    # Initialize the initial parameters:
    def __init__(self, alpha = 0.1, epochs = 100, gamma = 0.1, weights = None) -> None:
        self.alpha = alpha
        self.epochs = epochs
        self.gamma = gamma
        self.weights = weights

    # The Heaviside step function that we use as the activation function:
    def activation(self, x) -> int:
        return 1 if x >= 0 else 0

    def predict(self, X) -> int:
        # We add the 1 for the bias term as a constant to the X vector:
        X = np.insert(X, 0, 1)
        weighted_sum = np.dot(self.weights, X)
        return self.activation(weighted_sum)

    def fit(self, X, y) -> None:
        self.weights = np.zeros(X.shape[1] + 1)  # + 1 for the bias term

        # We run the training algorithm for a set number of epochs
        for epoch in range(self.epochs):
            total_error = 0
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)  # Add bias term
                weighted_sum = np.dot(self.weights, x_i)
                prediction = self.activation(weighted_sum)
                error = y[i] - prediction
                total_error += abs(error)
                self.weights += self.alpha * error * x_i  # Weight update function

            # We check if the termination condition is reached before the epochs are done:
            iteration_error = total_error / len(X)
            if iteration_error < self.gamma:
                print(f"Stopping early at epoch {epoch + 1}: iteration error {iteration_error} < gamma {self.gamma}")
                break

