import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from utils.hardware import Hardware
from utils.data import Data

class VRGestureRecognizer(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super(VRGestureRecognizer, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(256*4*7, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, num_classes)

        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        # Pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Convolutional part
        out = self.maxpool(self.relu(self.conv1(x)))
        out = self.maxpool(self.relu(self.conv2(out)))
        out = self.maxpool(self.relu(self.conv3(out)))
        out = self.maxpool(self.relu(self.conv4(out)))

        # Flattening feature maps
        out = torch.flatten(out, 1)

        # Fully connected part    
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.softmax(self.fc3(out))
        
        return out
    

    def compile(self, optimizer, loss_fn):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    def fit(self, X, y, epochs, batch_size, learning_rate) -> pd.DataFrame:
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        history = pd.DataFrame(columns=['Loss', 'Accuracy'])
        # Iterate over epochs
        for epoch in range(epochs):
            mean_loss = 0
            mean_acc = 0

            # Define a data generator
            data_gen = Data.data_generator(X, y, batch_size=batch_size)
            n_batches = 0
            self.train()
            for i, batch in enumerate(data_gen):
                n_batches += 1
                # Get batch data
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(Hardware.device()).unsqueeze(1), y_batch.to(Hardware.device())

                # Forward pass
                outputs = self(X_batch)
                loss = self.loss_fn(outputs, y_batch.long())
            
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update epoch loss
                mean_loss += loss.item()
                # Update epoch accuracy
                mean_acc += outputs.to('cpu').argmax(dim=1).eq(y_batch.to('cpu')).sum().item()

                # Compute mean loss and accuracy for the current epoch
            mean_loss /= n_batches
            mean_acc = mean_acc / (n_batches * batch_size)

            # Add epoch results to history
            history.loc[epoch] = [mean_loss, mean_acc]
        
            # Print epoch results
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.2f} | Accuracy: {mean_acc:.2f}")
            if mean_acc > 0.9:
                break
        
        return history
    
    def evaluate(self, X, y) -> tuple[float, float]:
        self.eval()
        with torch.no_grad():
            X = torch.from_numpy(X).to(Hardware.device()).unsqueeze(1)
            y = torch.from_numpy(y).to(Hardware.device())
            outputs = self(X)
            loss = self.loss_fn(outputs, y.long())
            acc = outputs.to('cpu').argmax(dim=1).eq(y.to('cpu')).sum().item() / X.shape[0]

            return loss, acc