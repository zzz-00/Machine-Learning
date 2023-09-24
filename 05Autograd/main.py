import torch


x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(
    5, 3, requires_grad=True
)  # You can set the value of requires_grad when creating a tensor, or later by using x.requires_grad_(True) method.
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
