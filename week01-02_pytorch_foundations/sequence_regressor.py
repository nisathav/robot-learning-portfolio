import torch
import torch.nn as nn #creaating neural networks
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sensor_dataset import SensorDataset

class SequenceRegressor(nn.Module):
    def __init__(self, window_size: int, hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(in_features=window_size, out_features=hidden_size)
        self.layer2 = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

if __name__ == "__main__": #was this file run directly, or was it imported by something else?, only activate when run directly
    torch.manual_seed(42)  # reproducibility
    """
    "random" numbers in code are different every time you 
    run your program. torch.manual_seed(42) tells PyTorch 
    "use the same starting point for randomness every time,
    " so all those random numbers — random weight starting 
    values, random shuffle order, random noise — come out 
    identical on every run.
    """
    # load the model
    model = SequenceRegressor(10,5)

    # loading the loss calculation function and update function
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # it handles different parameters with 

    # loading synthetic data
    timesteps = torch.linspace(0, 100, 1000)
    signal = torch.sin(timesteps) + torch.randn(1000) * 0.1  # noise mimics real sensor imperfection

    # passing the dataset into custom dataset
    dataset = SensorDataset(signal, window_size=10)
    # print(f"Dataset size: {len(dataset)}")

    # loading the data into dataloader for furthur batch processing
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    #windows, targets = next(iter(loader)) #it iterates through the batches and returns the next item form the iterator
    #print(f"Batch shapes: windows={windows}, targets={targets}")

    # --- training loop ---
    num_epochs = 5   # chose 5 to begin with 
    epoch_losses = [] 

    for epoch in range(num_epochs):
        model.train()   
        running_loss = 0.0
        for window, target in loader:
            # 1. forward
            outputs = model(window)
            # 2. loss
            loss = criterion(outputs.squeeze(), target)
            # 3. backward
            loss.backward()
            # 4. update
            optimizer.step()
            # 5. zero grad
            optimizer.zero_grad()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        epoch_losses.append(avg_loss)

        print(f"epoch {epoch}, loss {loss.item():.4f}")
        print(f"epoch {epoch}, avg loss {avg_loss:.4f}")

    #graph 
    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("loss_curve_sequence_regressor")
    plt.savefig("plots/loss_curve_sequence_regressor.png")
    print("Saved loss curve to plots/loss_curve_sequence_regressor.png")

    #evaluation
    model.eval()
    with torch.no_grad():
        sample_windows, sample_targets = next(iter(loader))
        sample_outputs = model(sample_windows).squeeze()

        for i in range(5):  # just look at first 5 samples in this batch
            print(f"window: {sample_windows[i]}")
            print(f"target: {sample_targets[i].item():.4f}, predicted: {sample_outputs[i].item():.4f}")
            print("---")
