from minitorch.engine import Tensor
import torch


def test_basic_add_tensors():
    a = Tensor(4)
    b = Tensor(3)
    c = a + b

    a_t = torch.tensor(4)
    b_t = torch.tensor(3)
    c_t = a_t + b_t
    
    print(f"c.data: {c.data}, c_t.data: {c_t.data}")

def test_grad_add_tensors():
    print(f"Running test_grad_add_tensors")
    a = Tensor(4)
    b = Tensor(3)
    c = a + b
    d = c + a
    d.backward()
    
    a_t = torch.tensor(4.0, requires_grad=True)
    b_t = torch.tensor(3.0, requires_grad=True)
    c_t = a_t + b_t
    c_t.retain_grad() # PyTorch only stores gradients on leaf tensors by default to save memory
    d_t = c_t + a_t
    d_t.retain_grad()
    d_t.backward()
    
    print(f"d.grad: {d.grad}, d_t.grad: {d_t.grad}")
    print(f"c.grad: {c.grad}, c_t.grad: {c_t.grad}")
    print(f"b.grad: {b.grad}, b_t.grad: {b_t.grad}")
    print(f"a.grad: {a.grad}, a_t.grad: {a_t.grad}")

def test_grad_mul_tensors():
    print("Running test_grad_mul_tensors")
    a = Tensor(4)
    b = Tensor(3)
    c = Tensor(2)
    d = a + b
    e = d * c
    e.backward()

    a_t = torch.tensor(4.0, requires_grad=True)
    b_t = torch.tensor(3.0, requires_grad=True)
    c_t = torch.tensor(2.0, requires_grad=True)
    d_t = a_t + b_t
    d_t.retain_grad() # PyTorch only stores gradients on leaf tensors by default to save memory
    e_t = d_t * c_t
    e_t.retain_grad()
    e_t.backward()

    print(f"e.grad: {e.grad}, e_t.grad: {e_t.grad}")
    print(f"d.grad: {d.grad}, d_t.grad: {d_t.grad}")
    print(f"c.grad: {c.grad}, c_t.grad: {c_t.grad}")
    print(f"b.grad: {b.grad}, b_t.grad: {b_t.grad}")
    print(f"a.grad: {a.grad}, a_t.grad: {a_t.grad}")

if __name__ == "__main__":
    test_grad_add_tensors()