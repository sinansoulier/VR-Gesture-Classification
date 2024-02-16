import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from vr_recognition.data import Data
from vr_recognition.hardware import Hardware
from vr_recognition.nets.ccvae.encoder import Encoder
from vr_recognition.nets.ccvae.decoder import Decoder

DEVICE = Hardware.device()

class CCVAE(nn.Module):
    """
    Class-Conditional Variational AutoEncoder class.
    """
    def __init__(self, hidden_dim: int = 128, latent_dim: int = 2, input_shape: tuple = (72, 114), nb_classes: int = 4):
        super(CCVAE, self).__init__()
        self.encoder = Encoder(hidden_dim=hidden_dim)
        self.decoder = Decoder(hidden_dim=hidden_dim, img_shape=input_shape)

        self.nb_features = hidden_dim * 2
        self.intermediate_shape = (math.ceil(input_shape[0] / (2 ** 4)), math.ceil(input_shape[1] / (2 ** 4)))

        intermediary_dim = self.nb_features * self.intermediate_shape[0] * self.intermediate_shape[1]

        # Fully connected layers
        self.fc1 = nn.Linear(intermediary_dim + nb_classes, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim + nb_classes, intermediary_dim)

        self.latent_dim = latent_dim
        self.nb_classes = nb_classes

    def reparametrize(self, mu, logvar):
        sigma = torch.exp(logvar * 0.5).to(DEVICE)
        epsilon = torch.randn_like(sigma).to(DEVICE)

        return mu + epsilon * sigma
    
    def encode(self, x, y):
        encoded = self.encoder(x)
        flattened_encoded = encoded.view(x.size(0), -1)
        
        flattened_encoded = torch.cat((flattened_encoded, y), dim=1)
        hidden = self.fc1(flattened_encoded)

        return hidden
    
    def decode(self, z, y):
        z = torch.cat((z, y), dim=1)
        z = self.fc2(z)

        reshaped_z = z.view(z.size(0),
                            self.nb_features,
                            self.intermediate_shape[0],
                            self.intermediate_shape[1])

        return self.decoder(reshaped_z)
    
    def forward(self, x, y):
        y_encoded = F.one_hot(y, num_classes=self.nb_classes).float().to(DEVICE)
        encoded = self.encode(x, y_encoded)

        mu, logvar = self.mu_layer(encoded), self.logvar_layer(encoded)

        z = self.reparametrize(mu, logvar)

        decoded = self.decode(z, y_encoded)

        return decoded, mu, logvar

    def loss_function(self, x, x_hat, mu, logvar):
        reconstruction_loss = nn.MSELoss()(x_hat, x)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return reconstruction_loss + kld_loss
    
    def sample(self, n):
        with torch.no_grad():
            y = torch.randint(0, self.nb_classes, (n, 1)).to(DEVICE)
            y_encoded = F.one_hot(y, num_classes=self.nb_classes).float().squeeze(1)
            z = torch.randn(n, self.latent_dim).to(DEVICE)
            decoded = self.decode(z, y_encoded)
        return decoded.cpu().detach().numpy(), y.squeeze(-1).cpu().detach().numpy()
    
    def fit(self, X_train, X_val, y_train, y_val ,epochs: int = 10, learning_rate: float = 1e-3, batch_size: int = 32):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        n_train_batches = X_train.shape[0] // batch_size
        n_val_batches = X_val.shape[0] // batch_size

        history = pd.DataFrame(columns=['loss', 'val_loss'])

        print('Training on {} batches, validating on {} batches'.format(n_train_batches, n_val_batches))

        for epoch in tqdm(range(epochs)):
            mean_loss = 0
            self.train()
            for x, y in Data.labeled_data_generator(X_train, y_train, batch_size=batch_size):
                x = x.to(DEVICE).unsqueeze(1)
                y = y.to(DEVICE)
                x_hat, mu, logvar = self(x, y)
                loss = self.loss_function(x, x_hat, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mean_loss += loss.item()
            
            mean_loss /= n_train_batches

            self.eval()
            with torch.no_grad():
                mean_val_loss = 0
                for x, y in Data.labeled_data_generator(X_val, y_val, batch_size=batch_size):
                    x = x.to(DEVICE).unsqueeze(1)
                    y = y.to(DEVICE)
                    x_hat, mu, logvar = self(x, y)
                    val_loss = self.loss_function(x, x_hat, mu, logvar)

                    mean_val_loss += val_loss.item()

                mean_val_loss /= n_val_batches
                print('epoch [{}/{}] loss: {:.2f} | validation Loss: {:.2f}                         '.format(epoch+1, epochs, mean_loss, mean_val_loss), end='\r')

            history.loc[epoch] = [mean_loss, mean_val_loss]
        
        self.history = history

    def evaluate(self, X_test, y_test, batch_size: int = 32):
        self.eval()
        with torch.no_grad():
            test_mean_loss = 0
            n_test_batches = X_test.shape[0] // batch_size
            for x, y in Data.labeled_data_generator(X_test, y_test, batch_size=batch_size):
                x = x.to(DEVICE).unsqueeze(1)
                y = y.to(DEVICE)
                x_hat, z_mu, z_logvar = self(x, y)
                test_mean_loss = self.loss_function(x_hat, x, z_mu, z_logvar)
                test_mean_loss += test_mean_loss.item()
        
        test_mean_loss = test_mean_loss / n_test_batches
        print(f"Test loss: {test_mean_loss:.4f}")
    
    def plot_history(self):
        plt.plot(self.history['loss'], label='loss')
        plt.plot(self.history['val_loss'], label='val_loss')
        plt.legend(['Loss', 'Validation Loss'])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()


    def to_device(self) -> nn.Module:
        """
        Move the model to the appropriate device (CPU or GPU).
        Returns:
            nn.Module: Model moved to the appropriate device.
        """
        return self.to(DEVICE)