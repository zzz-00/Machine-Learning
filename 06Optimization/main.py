import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import os


# Get current working directory path
directory_dir = os.getcwd()
dst_dir = os.path.join(directory_dir, "06Optimization", "data")
print(dst_dir)

training_data = datasets.FashionMNIST(
    root=dst_dir,
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root=dst_dir,
    train=False,
    download=True,
    transform=ToTensor(),
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()

# Hyperparameters
# Hyperparameters are adjustable parameters that let you control the model optimization process
# Different hyperparameter values can impact model training and convergence rates
learning_rate = 1e-3
batch_size = 64
epochs = 5

# Optimization loop
# Once we set our hyperparameters, we can then train and optimize our model with an optimization loop.
# Each iteration of the optimization loop is called an epoch.
# Each epoch consists of two main parts:
# The Train Loop - iterate over the training dataset and try to converge to optimal parameters.
# The Validation/Test Loop - iterate over the test dataset to check if model performance is improving.
# Initial the loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
# Optimization is the process of adjusting model parameters to reduce model error in each training step.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Inside the training loop, optimization happens in three steps:
# Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
# Backpropagate the prediction loss with a call to loss.backward(). PyTorch deposits the gradients of the loss w.r.t. each parameter.
# Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.
