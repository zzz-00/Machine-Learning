import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import os


# Get current working directory path
directory_dir = os.getcwd()
dst_dir = os.path.join(directory_dir, "03Transforms", "data")
print(dst_dir)

# All TorchVision datasets have two parameters -transform to modify the features and target_transform to modify the labels
# The FashionMNIST features are in PIL Image format, and the labels are integers.
# For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors
ds = datasets.FashionMNIST(
    root=dst_dir,
    train=True,
    download=True,
    transform=ToTensor(),  # ToTensor converts a PIL image or NumPy ndarray into a FloatTensor. and scales the image’s pixel intensity values in the range [0., 1.]
    target_transform=Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(
            dim=0, index=torch.tensor(y), value=1
        )  # It first creates a zero tensor of size 10 and calls scatter_ which assigns a value=1 on the index as given by the label y.
    ),
)
