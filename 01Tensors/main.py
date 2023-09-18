import torch
import numpy as np


# Initializing a Tensor
# Directly from data
# Tensors can be created directly from data. The data type is automatically inferred.
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# from a Numpy array
# Tensors can be created from NumPy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# from another tensor
# The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

# With random or constant values:
# shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor
shape = (
    2,
    3,
)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# Attributes of a Tensor
# Tensor attributes describe their shape, datatype, and the device on which they are stored.
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Operations on Tensors
# We move oue tensors to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# Standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First colum: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

# Joining tensors
t1 = torch.cat(
    [tensor, tensor, tensor], dim=1
)  # You can use torch.cat to concatenate a sequence of tensors along a given dimension.
print(f"t1: {t1}")


# Arithmetic operations
# This compute the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ''tensor.T'' returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(f"y1: \n {y1}")
print(f"y2: \n {y2}")
print(f"y3: \n {y3}")

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(f"z1: \n {z1}")
print(f"z2: \n {z2}")
print(f"z3: \n {z3}")

# Single-element tensors
# You can convert a one-element tensor to a Python numerical value using item()
agg = tensor.sum()
agg_item = agg.item()
print(f"agg_item: \n {agg_item}")
print(f"Type of agg_item: \n {type(agg_item)}")

# In-place operation
# Operations that store the result into the operand are called in-place. They are denoted by a _ suffix.
print(f"tensor: \n {tensor} \n")
tensor.add_(5)
print(f"after add_(5): \n {tensor} \n")

# Bridge with NumPy
# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
# Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# A change in the tensor reflects in the NumPy array.
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
