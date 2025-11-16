class SGD:
    def __init__(self, parameters, lr=1e-3) -> None:
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            param.grad.fill(0.0)

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad