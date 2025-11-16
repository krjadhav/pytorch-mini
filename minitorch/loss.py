import numpy as np
from minitorch.tensor import Tensor


class CrossEntropyLoss:
    def __init__(self, reduction="sum"):
        if reduction != "sum":
            raise NotImplementedError("Only reduction='sum' is currently supported.")
        self.reduction = reduction

    def __call__(self, input, target):
        if not isinstance(input, Tensor):
            input = Tensor(input)

        if isinstance(target, Tensor):
            target_np = target.data.astype(np.int64)
        else:
            target_np = np.array(target, dtype=np.int64)

        logits = input.data

        if logits.ndim != 2:
            raise ValueError("CrossEntropyLoss expects input of shape (N, C).")

        if target_np.ndim != 1:
            target_np = target_np.reshape(-1)

        N, C = logits.shape

        if target_np.shape[0] != N:
            raise ValueError("Target length must match batch size.")

        # Numerically stable log-softmax
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
        log_probs = shifted - np.log(sum_exp)

        # Select log-probabilities of the correct classes and sum (reduction='sum')
        correct_log_probs = log_probs[np.arange(N), target_np]
        loss_value = -correct_log_probs.sum()

        # Create output tensor that participates in the autograd graph
        out = Tensor(loss_value, (input,), "cross_entropy")

        def _backward():
            # Gradient of sum cross-entropy w.r.t. logits: softmax - one_hot
            softmax = exp_shifted / sum_exp
            grad_logits = softmax
            grad_logits[np.arange(N), target_np] -= 1.0

            # Chain with upstream gradient (scalar) from out.grad
            input.grad += grad_logits * out.grad

        out._backward = _backward
        return out