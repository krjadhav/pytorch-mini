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
    
    check_pass("c", c.data, c_t)

def test_basic_mul_tensors():
    print(f"Running test_basic_mul_tensors")
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a * b
    
    a_t = torch.tensor([1, 2, 3])
    b_t = torch.tensor([4, 5, 6])
    c_t = a_t * b_t
    
    check_pass("c", c.data, c_t.data)

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


def test_tanh():
    print("Running test_tanh")
    a = Tensor([0.1, 0.2, 0.3])
    b = Tensor([0.4, 0.5, 0.6])
    c = Tensor([0.2, 0.2, 0.2])
    d = a + b
    e = d * c
    f = e.tanh()
    g = f.sum()
    g.backward()
    
    a_t = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, requires_grad=True)
    b_t = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, requires_grad=True)
    c_t = torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32, requires_grad=True)
    d_t = a_t + b_t
    d_t.retain_grad()
    e_t = d_t * c_t
    e_t.retain_grad()
    f_t = e_t.tanh()
    f_t.retain_grad()
    g_t = f_t.sum()
    g_t.retain_grad()
    g_t.backward()

    check_pass("g", g.data, g_t.data)
    check_pass("g.grad", g.grad, g_t.grad)
    check_pass("f.grad", f.grad, f_t.grad)
    check_pass("e.grad", e.grad, e_t.grad)
    check_pass("d.grad", d.grad, d_t.grad)
    check_pass("c.grad", c.grad, c_t.grad)
    check_pass("b.grad", b.grad, b_t.grad)
    check_pass("a.grad", a.grad, a_t.grad)

def check_pass(label, minitorch_val, pytorch_val):
    # Check if the shapes of the arrays are equal
    if not minitorch_val.shape == pytorch_val.shape:
        print(f"{label}: ❌ FAIL: shape mismatch | Minitorch Value: {minitorch_val.shape} vs PyTorch Value: {pytorch_val.shape}") 
        return
    

    # Check if the values are close enough
    if not minitorch_val.shape == ():
        for _minitorch_val, _pytorch_val in zip(minitorch_val, pytorch_val):
            if not abs(_minitorch_val - _pytorch_val) < 1e-6:
                print(f"{label}: ❌ FAIL: value mismatch | Minitorch Value: {minitorch_val} vs PyTorch Value: {pytorch_val}")
                return
    
    if minitorch_val.shape == ():
        if not abs(minitorch_val.item() - pytorch_val.item()) < 1e-6:
            print(f"{label}: ❌ FAIL: value mismatch | Minitorch Value: {minitorch_val} vs PyTorch Value: {pytorch_val}")
            return

    print(f"{label}: ✅ PASS | Minitorch Value: {minitorch_val} vs PyTorch Value: {pytorch_val}")    
    

if __name__ == "__main__":
    # test_basic_add_tensors()
    # test_basic_mul_tensors()
    # test_mul_backward()
    test_tanh()