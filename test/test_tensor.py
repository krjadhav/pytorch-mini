import torch
from minitorch.tensor import Tensor
from mnist_util import load_mnist_data

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
    # Format minitorch value with trailing zeros like PyTorch
    if minitorch_val.shape == ():
        # Scalar
        mt_str = f"{minitorch_val.item():.2f}"
    else:
        # Array - format each element with 4 decimals
        formatted = [f"{x:.4f}" for x in minitorch_val.flatten()]
        mt_str = "[" + ", ".join(formatted) + "]"
    
    # Check if the shapes of the arrays are equal
    if not minitorch_val.shape == pytorch_val.shape:
        print(f"{label}: ❌ FAIL: shape mismatch | Minitorch Value: {mt_str} vs PyTorch Value: {pytorch_val}") 
        return
    
    # Check if the values are close enough
    if not minitorch_val.shape == ():
        for _minitorch_val, _pytorch_val in zip(minitorch_val, pytorch_val):
            if not abs(_minitorch_val - _pytorch_val) < 1e-6:
                print(f"{label}: ❌ FAIL: value mismatch | Minitorch Value: {mt_str} vs PyTorch Value: {pytorch_val}")
                return
    
    if minitorch_val.shape == ():
        if not abs(minitorch_val.item() - pytorch_val.item()) < 1e-6:
            print(f"{label}: ❌ FAIL: value mismatch | Minitorch Value: {mt_str} vs PyTorch Value: {pytorch_val}")
            return

    print(f"{label}: ✅ PASS | Minitorch Value: {mt_str} vs PyTorch Value: {pytorch_val}")

def test_backward_relu():
    print("Running test_backward_relu")
    a = Tensor([0.1, 0.2, 0.3])
    b = Tensor([0.4, 0.5, 0.6])
    c = Tensor([0.2, 0.2, 0.2])
    d = a + b
    e = d * c
    f = e.relu()
    g = f.sum()
    g.backward()

    a_t = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, requires_grad=True)
    b_t = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32, requires_grad=True)
    c_t = torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32, requires_grad=True)
    d_t = a_t + b_t
    d_t.retain_grad()
    e_t = d_t * c_t
    e_t.retain_grad()
    f_t = e_t.relu()
    f_t.retain_grad()
    g_t = f_t.sum()
    g_t.retain_grad()
    g_t.backward()

    check_pass("g", g.data, g_t.data)
    check_pass("g.grad", g.grad, g_t.grad)
    check_pass("f", f.data, f_t.data)
    check_pass("f.grad", f.grad, f_t.grad)
    check_pass("e.grad", e.grad, e_t.grad)
    check_pass("d.grad", d.grad, d_t.grad)
    check_pass("c.grad", c.grad, c_t.grad)
    check_pass("b.grad", b.grad, b_t.grad)
    check_pass("a.grad", a.grad, a_t.grad)

def test_backward_log():
    print("Running test_backward_log")
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = Tensor([2, 2, 2])
    d = a + b
    e = d * c
    f = e.log()
    g = f.sum()
    g.backward()

    a_t = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    b_t = torch.tensor([4, 5, 6], dtype=torch.float32, requires_grad=True)
    c_t = torch.tensor([2, 2, 2], dtype=torch.float32, requires_grad=True)
    d_t = a_t + b_t
    d_t.retain_grad()
    e_t = d_t * c_t
    e_t.retain_grad()
    f_t = e_t.log()
    f_t.retain_grad()
    g_t = f_t.sum()
    g_t.retain_grad()
    g_t.backward()

    check_pass("g", g.data, g_t.data)
    check_pass("g.grad", g.grad, g_t.grad)
    check_pass("f", f.data, f_t.data)
    check_pass("f.grad", f.grad, f_t.grad)
    check_pass("e.grad", e.grad, e_t.grad)
    check_pass("d.grad", d.grad, d_t.grad)
    check_pass("c.grad", c.grad, c_t.grad)
    check_pass("b.grad", b.grad, b_t.grad)
    check_pass("a.grad", a.grad, a_t.grad)


def test_flatten_mnist_x_train():
    print("Running test_flatten_mnist_x_train")
    X_train, Y_train, X_test, Y_test = load_mnist_data()
    mt_tensor = Tensor(X_train/255)
    mt_flat = mt_tensor.flatten(start_dim=1)
    pt_tensor = torch.tensor(X_train/255, dtype=torch.float32)
    pt_flat = pt_tensor.flatten(start_dim=1)
    print(f"After flatten: {mt_flat.data.shape} | pt_flat.shape: {pt_flat.shape}")
    mt_array = mt_flat.data
    if mt_array.shape == pt_flat.shape and torch.allclose(torch.tensor(mt_array, dtype=torch.float32), pt_flat, atol=1e-6):
        print(f"flatten(X_train) values ✅ PASS | shape: {mt_array.shape}")
    else:
        print("flatten(X_train) values ❌ FAIL")
        
def test_argmax():
    print("Running test_argmax")
    data = [[0.1, 0.9, 0.0],
            [0.31, 0.22, 0.47],
            [0.3, 0.3, 0.4]]
    mt_tensor = Tensor(data)
    mt_argmax = mt_tensor.argmax(dim=1, keepdim=True)

    pt_tensor = torch.tensor(data, dtype=torch.float32)
    pt_argmax = torch.argmax(pt_tensor, dim=1, keepdim=True)
 
    print(f"argmax minitorch: {mt_argmax} | pytorch: {pt_argmax}")

if __name__ == "__main__":
    # test_basic_add_tensors()
    # test_basic_mul_tensors()
    # test_mul_backward()
    # test_tanh()
    # test_backward_relu()
    # test_backward_log()
    # test_flatten_mnist_x_train()
    test_argmax()