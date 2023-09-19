# Dataset and Dataloaders
# Dataset stores the samples and their corresponding labels
# DataLoader wraps an iterable around the Dataset to enable easy access to the samples

# Loading a Dataset
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
import matplotlib.pyplot as plt
import os
import pandas as pd


# Get current working directory path
directort_dir = os.getcwd()
dst_dir = os.path.join(directort_dir, "02Data", "data")
print(dst_dir)

training_data = datasets.FashionMNIST(
    root=dst_dir,  # The path where the train/test data is stored
    train=True,  # training or test dataset
    download=True,  # downloads the data from the internet if it's not available at root
    transform=ToTensor(),  # transform and target_transform specify the feature and label transformations
)

test_data = datasets.FashionMNIST(
    root=dst_dir,
    train=False,
    download=True,
    transform=ToTensor(),
)

# Iterating and Visualizing the Dataset
# We can index Datasets manually like a list: training_data[index].
# We use matplotlib to visualize some samples in our training data
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# Creating a Custom Dataset for your files
# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
class CustomImageDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):  # The __init__ function is run once when instantiating the Dataset object.
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(
        self,
    ):  # The __len__ function returns the number of samples in our dataset.
        return len(self.img_labels)

    def __getitem__(
        self, idx
    ):  # The __getitem__ function loads and returns a sample from the dataset at the given index idx
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# Preparing your data for training with DataLoaders
# While training a model, we typically want to pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting
# and use Python’s multiprocessing to speed up data retrieval.
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Iterate through the DataLoader
# Display image and label
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
