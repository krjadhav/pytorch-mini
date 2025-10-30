import torch
from minitorch.tensor import Tensor

def test_basic_add_tensors():
    print(f"Running test_basic_add_tensors")
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    
    a_t = torch.tensor([1, 2, 3])
    b_t = torch.tensor([4, 5, 6])
    c_t = a_t + b_t
    
    print(f"c.data: {c.data}, c_t.data: {c_t.data}")
    check_pass(c.data, c_t.data)

def test_basic_mul_tensors():
    print(f"Running test_basic_mul_tensors")
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a * b
    
    a_t = torch.tensor([1, 2, 3])
    b_t = torch.tensor([4, 5, 6])
    c_t = a_t * b_t
    
    print(f"c.data: {c.data}, c_t.data: {c_t.data}")
    check_pass(c.data, c_t.data)

def test_mul_backward():
    print(f"Running test_mul_backward")
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = Tensor([2, 2, 2])
    d = a + b
    e = d * c
    e.sum().backward()    

    a_t = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    b_t = torch.tensor([4, 5, 6], dtype=torch.float32, requires_grad=True)
    c_t = torch.tensor([2, 2, 2], dtype=torch.float32, requires_grad=True)
    d_t = a_t + b_t
    d_t.retain_grad()
    e_t = d_t * c_t
    e_t.retain_grad()
    e_t.sum().backward()
    
    check_pass("e.grad", e.grad, e_t.grad)
    check_pass("d.grad", d.grad, d_t.grad)
    check_pass("c.grad", c.grad, c_t.grad)
    check_pass("b.grad", b.grad, b_t.grad)
    check_pass("a.grad", a.grad, a_t.grad)

def check_pass(label, minitorch_val, pytorch_val):
    # ok = abs(tensor_val - torch_val.data) < 1e-6
    print(f"{label} |  Minitorch Value: {minitorch_val} vs PyTorch Value: {pytorch_val}")


if __name__ == "__main__":
    # test_basic_add_tensors()
    # test_basic_mul_tensors()
    test_mul_backward()
    