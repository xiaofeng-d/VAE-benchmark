
# Import libraries
import torchvision                             # contains image datasets and many functions to manipulate images
import torchvision.transforms as transforms    # to normalize, scale etc the dataset
from torch.utils.data import DataLoader        # to load data into batches (for SGD)
from torchvision.utils import make_grid        # Plotting. Makes a grid of tensors
from torchvision.datasets import MNIST         # the classic handwritten digits dataset
import matplotlib.pyplot as plt                # to plot our images
import numpy as np

# Create Dataset object.s Notice that ToTensor() transforms images to pytorch tensors AND scales the pixel values to be within [0, 1]. Also, we have separate Dataset
# objects for training and test sets. Data will be downloaded to a folder called 'data'.
trainset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
testset  = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Create DataLoader objects. These will give us our batches of training and testing data.
batch_size = 100
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader  = DataLoader(testset, batch_size=batch_size, shuffle=True)

trainset

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import torch.nn as nn                          # Class that implements a model (such as a Neural Network)
import torch.nn.functional as F                # contains activation functions, sampling layers etc
import torch.optim as optim                    # For optimization routines such as SGD, ADAM, ADAGRAD, etc

e_hidden = 500        # Number of hidden units in the encoder. See AEVB paper page 7, section "Marginal Likelihood"
d_hidden = 500        # Number of hidden units in the decoder. See AEVB paper page 7, section "Marginal Likelihood"
latent_dim = 2        # Dimension of latent space. See AEVB paper, page 7, section "Marginal Likelihood"
learning_rate = 0.001 # For optimizer (SGD or Adam)
weight_decay = 1e-5   # For optimizer (SGD or Adam)
epochs = 50           # Number of sweeps through the whole dataset

class VAE(nn.Module):
    def __init__(self, latent_dim):
        """Variational Auto-Encoder Class"""
        super(VAE, self).__init__()
        # Encoding Layers
        self.latent_dim = latent_dim
        self.e_input2hidden = nn.Linear(in_features=784, out_features=e_hidden)
        self.e_hidden2mean = nn.Linear(in_features=e_hidden, out_features=self.latent_dim)
        self.e_hidden2logvar = nn.Linear(in_features=e_hidden, out_features=self.latent_dim)

        # Decoding Layers
        self.d_latent2hidden = nn.Linear(in_features=self.latent_dim, out_features=d_hidden)
        self.d_hidden2image = nn.Linear(in_features=d_hidden, out_features=784)

    def encoder(self, x):
        # Shape Flatten image to [batch_size, input_features]
        x = x.view(-1, 784)
        # Feed x into Encoder to obtain mean and logvar
        x = F.relu(self.e_input2hidden(x))
        return self.e_hidden2mean(x), self.e_hidden2logvar(x)

    def decoder(self, z):
        return torch.sigmoid(self.d_hidden2image(torch.relu(self.d_latent2hidden(z))))

    def forward(self, x):
        # Encoder image to latent representation mean & std
        mu, logvar = self.encoder(x)

        # Sample z from latent space using mu and logvar
        if self.training:
            z = torch.randn_like(mu).mul(torch.exp(0.5*logvar)).add_(mu)
        else:
            z = mu

        # Feed z into Decoder to obtain reconstructed image. Use Sigmoid as output activation (=probabilities)
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

# Loss
def vae_loss(image, reconstruction, mu, logvar):
  """Loss for the Variational AutoEncoder."""
  # Binary Cross Entropy for batch
  BCE = F.binary_cross_entropy(input=reconstruction.view(-1, 28*28), target=image.view(-1, 28*28), reduction='sum')
  # Closed-form KL Divergence
  KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return BCE - KLD



# Instantiate VAE with Adam optimizer
vae = VAE(latent_dim=latent_dim)
vae = vae.to(device)    # send weights to GPU. Do this BEFORE defining Optimizer
optimizer = optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=weight_decay)
vae.train()            # tell the network to be in training mode. Useful to activate Dropout layers & other stuff

# Train
losses = []

for epoch in range(epochs):
  # Store training losses & instantiate batch counter
  losses.append(0)
  number_of_batches = 0

  # Grab the batch, we are only interested in images not on their labels
  for images, _ in trainloader:
    # Save batch to GPU, remove existing gradients from previous iterations
    images = images.to(device)
    optimizer.zero_grad()

    # Feed images to VAE. Compute Loss.
    reconstructions, latent_mu, latent_logvar = vae(images)
    loss = vae_loss(images, reconstructions, latent_mu, latent_logvar)

    # Backpropagate the loss & perform optimization step with such gradients
    loss.backward()
    optimizer.step()

    # Add loss to the cumulative sum
    losses[-1] += loss.item()
    number_of_batches += 1

  # Update average loss & Log information
  losses[-1] /= number_of_batches
  print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, epochs, losses[-1]))

# To save your trained model to Google Drive using PyDrive, you need to make some modifications to the code you've provided. Specifically, you'll need to save the model's state dictionary to a file and then upload this file to Google Drive. Here's how you can do it:
import torch

# After your training loop, save the model's state_dict
model_save_path = '/content/vae_model.pth'  # Temporary path in Google Colab
torch.save(vae.state_dict(), model_save_path)




# # Fetch a batch of test images
# test_images, _ = next(iter(testloader))  # Ignore the labels

# # Display original images
# with torch.no_grad():
#     print("Original Images")
#     test_images = test_images.cpu()
#     test_images = test_images.clamp(0, 1)
#     test_images = test_images[:50]
#     test_images = make_grid(test_images, 10, 5)
#     test_images = test_images.numpy()
#     test_images = np.transpose(test_images, (1, 2, 0))
#     plt.imshow(test_images)
#     plt.show()

# Display original images
with torch.no_grad():
    print("Original Images")
    images = images.cpu()
    images = images.clamp(0, 1)
    images = images[:50]
    images = make_grid(images, 10, 5)
    images = images.numpy()
    images = np.transpose(images, (1, 2, 0))
    plt.imshow(images)
    plt.show()

# Display reconstructed images
with torch.no_grad():
    print("Reconstructions")
    reconstructions = reconstructions.view(reconstructions.size(0), 1, 28, 28)
    reconstructions = reconstructions.cpu()
    reconstructions = reconstructions.clamp(0, 1)
    reconstructions = reconstructions[:50]
    plt.imshow(np.transpose(make_grid(reconstructions, 10, 5).numpy(), (1, 2, 0)))
    plt.show()



# """Evaluation"""

# # Set VAE to evaluation mode (deactivates potential dropout layers)
# # So that we use the latent mean and we don't sample from the latent space
# vae.eval()

# # Keep track of test loss (notice, we have no epochs)
# test_loss, number_of_batches = 0.0, 0

# for test_images, _ in testloader:

#   # Do not track gradients
#   with torch.no_grad():

#     # Send images to the GPU/CPU
#     test_images = test_images.to(device)

#     # Feed images through the VAE to obtain their reconstruction & compute loss
#     reconstructions, latent_mu, latent_logvar = vae(test_images)
#     loss = vae_loss(test_images, reconstructions, latent_mu, latent_logvar)

#     # Cumulative loss & Number of batches
#     test_loss += loss.item()
#     number_of_batches += 1

# # Now divide by number of batches to get average loss per batch
# test_loss /= number_of_batches
# print('average reconstruction error: %f' % (test_loss))



# vae.eval()
# with torch.no_grad():
#     # Sample from standard normal distribution
#     z = torch.randn(50, latent_dim, device=device)

#     # Reconstruct images from sampled latent vectors
#     recon_images = vae.decoder(z)
#     recon_images = recon_images.view(recon_images.size(0), 1, 28, 28)
#     recon_images = recon_images.cpu()
#     recon_images = recon_images.clamp(0, 1)

#     # Plot Generated Images
#     plt.imshow(np.transpose(make_grid(recon_images, 10, 5).numpy(), (1, 2, 0)))

# with torch.no_grad():
#   # Create empty (x, y) grid
#   latent_x = np.linspace(-1.5, 1.5, 20)
#   latent_y = np.linspace(-1.5, 1.5, 20)
#   latents = torch.FloatTensor(len(latent_x), len(latent_y), 2)
#   # Fill up the grid
#   for i, lx in enumerate(latent_x):
#     for j, ly in enumerate(latent_y):
#       latents[j, i, 0] = lx
#       latents[j, i, 1] = ly
#   # Flatten the grid
#   latents = latents.view(-1, 2)
#   # Send to GPU
#   latents = latents.to(device)
#   # Find their representation
#   reconstructions = vae.decoder(latents).view(-1, 1, 28, 28)
#   reconstructions = reconstructions.cpu()
#   # Finally, plot
#   fig, ax = plt.subplots(figsize=(10, 10))
#   plt.imshow(np.transpose(make_grid(reconstructions.data[:400], 20, 5).clamp(0, 1).numpy(), (1, 2, 0)))

