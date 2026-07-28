- The need for custom dataset, because the sensor data do not come with a convenient torchvision.datasets wrapper
- Any custom dataset is a python class that inherits from torch.utils.data.Dataset and they implement two methods;
    1. __len__(self) - returns the total number of samples are there in the dataset
    2. __getitem__(self, idx) - given the index it returns ona sample from the data along with its label/target

    ** mostly DataLoader uses these function to create the batches and to assemble the batch when needed

Applications: given the last N readings it should predict the next reading within a certian loss

Question: if you have 1000 total timesteps and you pick a window size of, say, 10, roughly how many valid training samples can you extract from that single sequence? Think about it as: at which positions in the array can you both (a) look back far enough to grab a full window of 10, and (b) still have one more reading after that window to use as the target?

Take a guess at the count, and also tell me: do you think consecutive samples (e.g., the window starting at index 0 vs. index 1) should share most of their data (heavily overlapping windows), or should each sample be a completely separate, non-overlapping chunk of the signal? Reason about which choice gives you more training data to work with.

Answer: 
- each training sample needs a window of 10 consecutive values plus one more as target. so, a window looks i to i+10,
        i + 10 <= 999
        i <= 989
- so total number of samples available for training is 990

- overlapping windows are better because it donot yield more training sample but obtaining data in real life is expensive.