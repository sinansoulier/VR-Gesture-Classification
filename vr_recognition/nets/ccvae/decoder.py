import torch
import torch.nn as nn

class Trim(nn.Module):
    """
    Class that defines a trimming layer.
    """
    def __init__(self, size1: int, size2: int):
        super(Trim, self).__init__()
        self.size1 = size1
        self.size2 = size2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer. This method is called when performing inference, training or evaluation.
        Params:
            x (torch.Tensor): Input data
        Returns:
            torch.Tensor: Trimmed data.
        """
        return x[:, :, :self.size1, :self.size2]


class Decoder(nn.Module):
    """
    Class defining the decoder architecture of the VAE.
    """
    def __init__(self, hidden_dim: int = 2, img_shape: tuple[int, int] = (72, 114)):
        super(Decoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv4 = nn.ConvTranspose2d(hidden_dim, 1, kernel_size=3, stride=2, padding=1, output_padding=0)

        # Activation functions
        self.relu = nn.LeakyReLU(0.1)

        # Trimming
        self.trim = Trim(img_shape[0], img_shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Decoded tensor representing the reconstructed data.
        """
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        decoded = self.relu(self.deconv4(x))

        decoded = self.trim(decoded)
        
        return decoded