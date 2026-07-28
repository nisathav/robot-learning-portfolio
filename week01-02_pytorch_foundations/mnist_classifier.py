import torch
import torch.nn as nn //creaating neural networks
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(42)

# --- data (from before) ---
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# --- model ---
class MyNet(nn.Module): #creating the neural network model including the number of layer, input translation and layer interonnection
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=784, out_features=128)
        self.layer2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

model = MyNet()

# --- loss + optimizer ---

# CrossEntropyLoss combines softmax + negative log-likelihood in one step.
# It takes the model's raw logits (10 unbounded scores per MNIST image, one
# per digit class) and internally converts them into a probability
# distribution via softmax: p_i = exp(z_i) / sum(exp(z_j)). It then computes
# loss = -log(p_true), the negative log-probability assigned to the correct
# digit. Confident + correct predictions push p_true near 1, so loss is
# near 0; confident + wrong predictions push p_true near 0, so loss blows
# up toward infinity. That steep penalty on confident mistakes is what
# produces large gradients for bad predictions and small gradients for
# good ones, driving the weight updates each step.
#
# IMPORTANT: pass raw logits here, NOT pre-softmaxed probabilities -
# PyTorch applies softmax internally (via log_softmax + nll_loss, for
# numerical stability). Also pass labels as class indices (LongTensor,
# shape [batch], values 0-9), not one-hot vectors.
criterion = nn.CrossEntropyLoss() 
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1)   # this iterates through the tuple with model parameters and help calculate the gradient decesent
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- training loop ---
num_epochs = 5   # chose 5 to begin with 
# larger number not going to work, 64 per batch so for epoch 60,000 / 64 = 934, times 100 takes 93,400 cross flows

epoch_losses = []

for epoch in range(num_epochs):
    model.train()   
    running_loss = 0.0
    for images, labels in train_loader:
        # 1. forward
        outputs = model(images)

        # 2. loss
        loss = criterion(outputs, labels)

        # 3. backward
        loss.backward()

        # 4. update
        optimizer.step()

        # 5. zero grad
        optimizer.zero_grad()

        running_loss += loss.item()

    #total loss summed across all batches, divided by how many batches contributed to that sum.
    avg_loss = running_loss / len(train_loader)

    epoch_losses.append(avg_loss)
    print(f"epoch {epoch}, loss {loss.item():.4f}")
    print(f"epoch {epoch}, avg loss {avg_loss:.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        """
        dim=1 collapses across the class-score axis within each row, giving you one predicted class 
        per image (exactly batch_size predictions), which is what you want. dim=0 would have done 
        the opposite — compared scores down each column across different images, 
        which is meaningless here

        dim=0, gives the answer to the which image in the batch
        """
        predictions = torch.argmax(outputs, dim=1)  
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Validation accuracy: {accuracy*100:.2f}%")

"""
the assignments needs accuracy more than 97%, so doing the following changes but one by one,
    1. changing the epoch range from 5 to 15
        effect on the output - it worked accuracy above 97.97%
    2. try changing the torch.optim.SGD to torch.optim.Adam got 97.04%
"""
plt.plot(epoch_losses)
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("MNIST Training Loss")
plt.savefig("plots/loss_curve.png")
print("Saved loss curve to plots/loss_curve.png")

