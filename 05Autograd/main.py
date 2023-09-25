import torch


x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(
    5, 3, requires_grad=True
)  # You can set the value of requires_grad when creating a tensor, or later by using x.requires_grad_(True) method.
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# Computing Gradients
loss.backward()
print(w.grad)
print(b.grad)

# Disabling Gradient Tracking
# By default, all tensors with requires_grad=True are tracking their computational history and support gradient computation.
# However, there are some cases when we do not need to do that,
# for example, when we have trained the model and just want to apply it to some input data, i.e. we only want to do forward computations through the network.
# We can stop tracking computations by surrounding our computation code with torch.no_grad() block:
z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

# Another way to achieve the same result is to use the detach() method on the tensor:
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

inp = torch.eye(4, 5, requires_grad=True)
out = (inp + 1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
