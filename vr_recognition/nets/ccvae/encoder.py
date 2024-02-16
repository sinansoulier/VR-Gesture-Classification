import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Class defining the encoder architecture of the Variational Autoencoder.
    """
    def __init__(self, hidden_dim: int = 2):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1)

        # Activation functions
        self.relu = nn.LeakyReLU(0.1)

        # Dropout layer(s)
        self.dropout = nn.Dropout(p=0.3)
        
        # Pooling layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model. This method is called when performing inference, training or evaluation.
        Params:
            x (torch.Tensor): Input data
        Returns:
            torch.Tensor: Output data.
                          The shape of the output tensor depends on the number of classes that characterizes the model.
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        encoded = self.relu(self.conv4(x))

        encoded = self.dropout(encoded)
        return encoded