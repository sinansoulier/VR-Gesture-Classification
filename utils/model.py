import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from utils.hardware import Hardware
from utils.data import Data

class VRGestureRecognizer(nn.Module):
    """
    Class that defines the model architecture. This class inherits from nn.Module,
    which is the base class for all neural network modules in PyTorch.

    The model architecture is composed of 4 convolutional layers and 3 fully connected layers.

    The training loop, evaluation and export to ONNX are also defined in this class for convenience.
    """

    def __init__(self, hidden_size: int, num_classes: int):
        """
        Constructor of the class. Here we define the model architecture by instantiating the layers.
        Params:
            hidden_size (int): Size of the hidden layer(s)
            num_classes (int): Number of classes characterizing the dataset.
                               It is used to define the output layer.
        """
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

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model. This method is called when performing inference, training or evaluation.
        Params:
            x (torch.Tensor): Input data
        Returns:
            torch.Tensor: Output data.
                          The shape of the output tensor depends on the number of classes that characterizes the model.
        """
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
        """
        Compile the model by defining the optimizer and the loss function.
        Params:
            optimizer: Optimizer to use for training
            loss_fn: Loss function to use for training
        """
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    def fit(self, X, y, epochs, batch_size, learning_rate) -> pd.DataFrame:
        """
        Train the model on a given dataset and from hyperparameters.
        Params:
            X (np.ndarray): Training data.
            y (np.ndarray): Training labels (ground truth).
            epochs (int): Number of epochs to train the model.
            batch_size (int): Size of the batch.
            learning_rate (float): Learning rate.
        Returns:
            pd.DataFrame: Training history.
        """
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
        
        self.history = history
        
        return history
    
    def evaluate(self, X, y) -> tuple[float, float]:
        """
        Evaluate the model on a given dataset.
        Params:
            X (np.ndarray): Evaluation data.
            y (np.ndarray): Evaluation labels (ground truth).
        Returns:
            tuple[float, float]: Tuple containing the loss and the accuracy.
        """
        self.eval()
        with torch.no_grad():
            X = torch.from_numpy(X).to(Hardware.device()).unsqueeze(1)
            y = torch.from_numpy(y).to(Hardware.device())
            outputs = self(X)
            loss = self.loss_fn(outputs, y.long())
            acc = outputs.to('cpu').argmax(dim=1).eq(y.to('cpu')).sum().item() / X.shape[0]

            return loss, acc
    
    def plot_history(self) -> None:
        """
        Plot the training history. The history must be a dataframe with the following columns:
            - Loss
            - Accuracy
        Params:
            history (pd.DataFrame): Training history
        """
        _, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(self.history['Loss'], color='red')
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')

        ax[1].plot(self.history['Accuracy'])
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')

        plt.show()

    def export_to_onnx(self, path: str) -> None:
        """
        Export the a model instance - with all the weights and biases - to ONNX format.
        This generated file can be used to run inference on a device or environment that does not support PyTorch,
        such as a mobile device or a Virtual Reality headset.
        Params:
            path (str): Path to the ONNX file
        """
        self.eval()
        X = torch.randn(1, 1, 71, 114, requires_grad=True)
        torch.onnx.export(
            self.to('cpu'),
            X,
            path,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )