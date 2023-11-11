import tarfile
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from colorizers.dataset import Dataset

import trainer


# Number of epochs in training
epochs = 1000

# Learning rate
learning_rate = 0.01

# Load samples and targets (I redefined the cifar-100 dataset - removed everything but the images themselves)
samples = np.load("../datasets/cifar-100-python/train_grayscale")
targets = np.load("../datasets/cifar-100-python/train")

# Convert data to tensors
samples = torch.from_numpy(samples[0:50])
samples = samples.to(torch.int)
targets = torch.from_numpy(targets[0:50])
targets = targets.to(torch.int)

# Put samples and targets into a dataset
dataset = Dataset(samples, targets)
train_dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

# Create a model from network
model = Network().to('cuda:0')  #TODO: define/create a neural network

# Create a trainer
trainer = trainer.Trainer(
    model=model,
    dataset=dataset,
    loss_fn=nn.MSELoss(),
    optimizer=optim.Adam(model.parameters(), lr=learning_rate)  #TODO: choose optimizer (I just chose one of the more common ones for the time being)
)

# Train the model
trainer.train(epochs=epochs)

# Save the trained model
torch.save(model.state_dict(), "trained_model.pt")


# Check if prediction is working
'''data = np.load("../datasets/cifar-100-python/test_grayscale")

model = Network()
model.load_state_dict(torch.load("trained_model.pt"))
input_data = torch.from_numpy(data[0])
input_data = input_data.to(torch.float32)
prediction = model(input_data).cpu()
image = prediction.detach().numpy()

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
#image = data[0]
#image = util.grayscale(image)
image = image.reshape(3,32,32).transpose(1,2,0)
plt.imshow(image)
plt.show()'''