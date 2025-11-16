class SGD:
    """Stochastic Gradient Descent"""
    def __init__(self, parameters, lr=1e-3) -> None:
        # TODO: Add more parameters like momentum later
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        # Reset gradients. This is needed because gradients are accumulated
        # and will result in incorrect gradient updates and suboptimal model training.
        for param in self.parameters:
            param.grad.fill(0.0)

    def step(self):
        # Update parameters using gradient descent
        for param in self.parameters:
            param.data -= self.lr * param.grad