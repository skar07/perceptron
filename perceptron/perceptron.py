import numpy as np

class Perceptron:
    """Simple perceptron classifier implementation inspired by Rosenblatt's algorithm
    
        The net input function is acquired, and then threshold function is used to
        predict an output.
        Error is calculated if output is not equal to the target and appended to the errors array.
        The weights and the bias size are updated with each epoch.
    """
    def __init__(self, eta=0.05, n_iter=100, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self, X, y):
        random_gen = np.random.RandomState(self.random_state)
        self.w_ = random_gen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float(0.)
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        
    
