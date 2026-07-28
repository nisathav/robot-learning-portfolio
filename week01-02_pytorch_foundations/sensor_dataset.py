import torch
from torch.utils.data import Dataset, DataLoader

class SensorDataset(Dataset):
    def __init__(self, signal: torch.Tensor, window_size: int):
        self.signal = signal
        self.window_size = window_size

    def __len__(self):
        # how many valid (window, target) pairs exist?
        # use the exact reasoning you just worked through above
        number = (len(self.signal) - self.window_size)
        return number

    def __getitem__(self, idx):
        # window: window_size values starting at idx
        window = self.signal[idx:idx+self.window_size]
        # target: the single value right after the window
        target = self.signal[idx+self.window_size]
        return window, target

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

    # loading synthetic data
    timesteps = torch.linspace(0, 100, 1000)
    signal = torch.sin(timesteps) + torch.randn(1000) * 0.1  # noise mimics real sensor imperfection

    # passing the dataset into custom dataset
    dataset = SensorDataset(signal, window_size=10)
    print(f"Dataset size: {len(dataset)}")

    # loading the data into dataloader for furthur batch processing
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    windows, targets = next(iter(loader)) #it iterates through the batches and returns the next item form the iterator
    print(f"Batch shapes: windows={windows.shape}, targets={targets.shape}")