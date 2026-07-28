#loading data set
import torchvision #toolset for computer vision
import torchvision.transforms as transforms # contain functions that modify images
from torch.utils.data import DataLoader # feeds data to neural network efficiently

# converts the raw PIL images into PyTorch tensors and scales pixel values from 0–255 
# into 0.0–1.0. Neural nets train 
# much better on small, normalized numbers than raw 0–255 integers
transform = transforms.ToTensor() #PIL image to PyTorch Tensor

# Download the dataset, downloades the 60,000 training images
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
# Download the validation dataset, 10,000 unseen images
val_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# creates batches od images from 60,000 in 64 suffled
"""
one complete pass through all batches is called an epoch

60,000 / 64 = 938 batches
"""
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

"""
it iterates through the batches and returns the 
next item form the iterator
"""
images, labels = next(iter(train_loader))
print(images.shape, labels.shape)

"""
output 
image
torch.size([64,1,28,28])
torch.size([no.of.images ina batch, number of channels of the color, height, width])

labels
torch.size([64])
each image represented by one number
"""

"""
MNIST Dataset
(70,000 images)
          │
          ▼
 torchvision.datasets.MNIST
          │
          ▼
 transforms.ToTensor()
(PIL → Tensor, 0–255 → 0–1)
          │
          ▼
 DataLoader
(batch_size=64)
          │
          ▼
 Batch 1
(64 images, 64 labels)
          │
          ▼
 images.shape = (64, 1, 28, 28)
 labels.shape = (64)
          │
          ▼
 Feed into the neural network
"""
