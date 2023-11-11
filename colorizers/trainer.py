import torch

class Trainer:
    def __init__(self, model, dataset, optimizer, loss_fn):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def train(self, epochs):
        learning_rate = self.optimizer.param_groups[0]['lr']

        for epoch in range(epochs):
            # Train on the training data
            for batch_idx, (samples, targets) in enumerate(self.dataset):
                # Forward pass
                predictions = self.model(samples.to(self.device))
                #predictions = self.model(samples)  # without GPU

                # Calculate the loss
                loss = self.loss_fn(predictions.to(self.device), targets.to(self.device))
                #loss = self.loss_fn(predictions, targets)  # without GPU

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Learning rate that decays linearly (comment out this block if you want a constant learning rate)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = learning_rate - (learning_rate / epochs) * epoch

                # Print the loss
                if batch_idx % 100 == 0:
                    print('Epoch: {} | Batch: {} | Loss: {}'.format(epoch, batch_idx, loss.item()))