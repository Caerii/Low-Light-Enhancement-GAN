# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader


# Custom Dataset class for loading the superresolution dataset
class SuperResolutionDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = os.listdir(self.root)
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.image_files[idx])
        image = default_loader(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image


# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {DEVICE} backend")

# Training data
DATASET_ROOT = "super_dataset/high"  # Path to your superresolution dataset folder

# Batch size for training models
BATCH_SIZE = 128

# Image size
IMG_SIZE = 2500

# Number of training epochs
NUM_EPOCHS = 20

# Size of latent vector z
SIZE_Z = 100

# Number of discriminator steps for each generator step
K_STEPS = 1

# Learning rate for Adam optimizer
ADAM_LR = 0.0002

# Beta1 hyperparameter for Adam optimizer
ADAM_BETA1 = 0.5

# Create transform to resize and normalize images
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Create custom dataset and data loader
dataset = SuperResolutionDataset(DATASET_ROOT, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Generator Model (G)
class Generator(nn.Module):
    def __init__(self, input_size=SIZE_Z):
        super(Generator, self).__init__()

        self.layer_x = nn.Sequential(
            nn.ConvTranspose2d(input_size, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer_y = nn.Sequential(
            nn.ConvTranspose2d(10, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer_xy = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.layer_x(x)
        y = self.layer_y(y)
        xy = torch.cat([x, y], dim=1)
        xy = self.layer_xy(xy)
        return xy


# Discriminator Model (D)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer_x = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_y = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer_xy = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = self.layer_x(x)
        y = self.layer_y(y)
        xy = torch.cat([x, y], dim=1)
        xy = self.layer_xy(xy)
        xy = xy.view(xy.size(0), -1)
        return xy


# Create the Generator and Discriminator models
netG = Generator().to(DEVICE)
netD = Discriminator().to(DEVICE)

# Initialize the weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)

# Binary cross entropy loss
criterion = nn.BCELoss()

# Adam optimizers for Generator and Discriminator
optimizerG = torch.optim.Adam(netG.parameters(), lr=ADAM_LR, betas=(ADAM_BETA1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=ADAM_LR, betas=(ADAM_BETA1, 0.999))

# Labels for training images x for Discriminator training
labels_real = torch.ones((BATCH_SIZE, 1)).to(DEVICE)
# Labels for generated images G(z) for Discriminator training
labels_fake = torch.zeros((BATCH_SIZE, 1)).to(DEVICE)

# Fixed noise for testing generator and visualization
z_test = torch.randn(100, SIZE_Z).to(DEVICE)

# Convert labels to one-hot encoding
onehot = torch.zeros(10, 10).scatter_(1, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1)
fill = torch.zeros([10, 10, IMG_SIZE, IMG_SIZE])
for i in range(10):
    fill[i, i, :, :] = 1
test_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10).type(torch.LongTensor)
test_Gy = onehot[test_y].to(DEVICE)

# Lists for storing loss and visualization values
D_losses = []
G_losses = []
Dx_values = []
DGz_values = []

# Training loop
step = 0
for epoch in range(NUM_EPOCHS):
    epoch_D_losses = []
    epoch_G_losses = []
    epoch_Dx = []
    epoch_DGz = []

    for images in dataloader:
        step += 1
        x = images.to(DEVICE)
        y_labels = torch.randint(0, 10, (BATCH_SIZE,)).to(DEVICE)  # Generate random label tensor

        D_y = fill[y_labels].to(DEVICE)

        x_preds = netD(x, D_y)
        D_x_loss = criterion(x_preds, labels_real)

        z = torch.randn(BATCH_SIZE, SIZE_Z).to(DEVICE)
        y_gen = (torch.rand(BATCH_SIZE, 1) * 10).type(torch.LongTensor).squeeze()
        G_y = onehot[y_gen].to(DEVICE)
        DG_y = fill[y_gen].to(DEVICE)

        fake_image = netG(z, G_y)
        z_preds = netD(fake_image.detach(), DG_y)
        D_z_loss = criterion(z_preds, labels_fake)

        D_loss = D_x_loss + D_z_loss
        epoch_D_losses.append(D_loss.item())
        epoch_Dx.append(x_preds.mean().item())

        netD.zero_grad()
        D_loss.backward()
        optimizerD.step()

        if step % K_STEPS == 0:
            z_out = netD(fake_image, DG_y)
            G_loss = criterion(z_out, labels_real)
            epoch_DGz.append(z_out.mean().item())
            epoch_G_losses.append(G_loss)

            netG.zero_grad()
            G_loss.backward()
            optimizerG.step()

    D_losses.append(sum(epoch_D_losses) / len(epoch_D_losses))
    G_losses.append(sum(epoch_G_losses) / len(epoch_G_losses))
    Dx_values.append(sum(epoch_Dx) / len(epoch_Dx))
    DGz_values.append(sum(epoch_DGz) / len(epoch_DGz))

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Discriminator Loss: {D_losses[-1]:.3f} Generator Loss: {G_losses[-1]:.3f}"
          f" D(x): {Dx_values[-1]:.3f} D(G(x)): {DGz_values[-1]:.3f}")

    # Generate images after each epoch and save
    netG.eval()
    with torch.no_grad():
        fake_test = netG(z_test, test_Gy).cpu()
        torchvision.utils.save_image(fake_test, f"superresolution_epoch_{epoch+1}.jpg", nrow=10, padding=0, normalize=True)
    netG.train()

# Load saved generated images grid and visualize using matplotlib
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("Generated Images")
_ = plt.imshow(Image.open(f"superresolution_epoch_{NUM_EPOCHS}.jpg"))

# Plot Discriminator and Generator loss over the epochs
plt.figure(figsize=(10, 5))
plt.title("Discriminator and Generator Loss during Training")
plt.plot(D_losses, label="D Loss")
plt.plot(G_losses, label="G Loss")
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel("Num Epochs")
plt.legend()
plt.show()
