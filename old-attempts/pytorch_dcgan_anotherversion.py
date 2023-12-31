# -*- coding: utf-8 -*-
"""Pytorch_DCGAN_AnotherVersion.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16XstbkDopABbUUr41sq-Sy9HABte23DW
"""

from google.colab import drive
# drive.mount('/content/gdrive')
drive.mount('/content/gdrive', force_remount=True)

!ls gdrive/MyDrive/'Computational Imaging Project - DTU 34269 - Low Light Super Resolution'

# !ls gdrive/MyDrive/'Computational Imaging Project - DTU 34269 - Low Light Super Resolution'/dataset3
!ls gdrive/MyDrive/'Computational Imaging Project - DTU 34269 - Low Light Super Resolution'/dataset3

# #path that contains folder you want to copy
# %cd gdrive/MyDrive/'Computational Imaging Project - DTU 34269 - Low Light Super Resolution'/
# # %cp -av YOUR_FOLDER NEW_FOLDER_COPY dataset3
# %cp -av dataset3 dataset3_cpy

# !gdown --folder gdrive/MyDrive/'Computational Imaging Project - DTU 34269 - Low Light Super Resolution'/dataset3

# !unzip gdrive/MyDrive/'Computational Imaging Project - DTU 34269 - Low Light Super Resolution'/Data.zip

# "path/to/dataset" /Users/jamesau/Library/CloudStorage/GoogleDrive-jamesau2810@gmail.com/.shortcut-targets-by-id/1D_3mIrLJdP9jl6X_I0o0Bwd83-TnZKZT/Computational Imaging Project - DTU 34269 - Low Light Super Resolution/dataset3
# Root directory for dataset
# For new Dataset Computational Imaging Project - DTU 34269 - Low Light Super Resolution/dataset3/super_dataset
dataroot = "gdrive/MyDrive/Computational Imaging Project - DTU 34269 - Low Light Super Resolution/dataset3/super_dataset"
# Original Zipped Data
# dataroot = "Data"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
# Define the generator network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Define the discriminator network
class Discriminator(nn.Module):
    # def __init__(self, ngpu):
    def __init__(self):
        super(Discriminator, self).__init__()
        # self.ngpu = ngpu
        # Number of channels in the training images. For color images this is 3
        nc = 3
        # Size of feature maps in discriminator
        ndf = 64
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    # def __init__(self):
    #     super(Discriminator, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    #     self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
    #     self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    #     self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
    #     self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    #     self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
    #     self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    #     self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
    #     # self.fc1 = nn.Linear(512 * 8 * 8, 1024)
    #     # self.fc2 = nn.Linear(1024, 1)
    #     self.fc1 = nn.Linear(512 * 8 * 2, 1024)
    #     self.fc2 = nn.Linear(1024, 1)
    #     self.leakyrelu = nn.LeakyReLU(0.2)

    # def forward(self, x):
    #     x = self.leakyrelu(self.conv1(x))
    #     x = self.leakyrelu(self.conv2(x))
    #     x = self.leakyrelu(self.conv3(x))
    #     x = self.leakyrelu(self.conv4(x))
    #     x = self.leakyrelu(self.conv5(x))
    #     x = self.leakyrelu(self.conv6(x))
    #     x = self.leakyrelu(self.conv7(x))
    #     x = self.leakyrelu(self.conv8(x))
    #     # print(np.shape(x))
    #     x = x.view(x.size(0), -1)
    #     # x = x.view(x.size(0)//4,-1)
    #     # print(np.shape(x))
    #     x = self.leakyrelu(self.fc1(x))
    #     x = self.fc2(x)
    #     return x

# Define the training loop
Blur_size = 4

def train(generator, discriminator, dataloader, optimizer_g, optimizer_d, criterion):
    for epoch in range(num_epochs):
        # For the low resolution data already provided
        for i, (low_res, high_res) in enumerate(dataloader):
            # end
            # For creation of low data with downsampling
            # for i, data in enumerate(dataloader):
            # high_res = data
            # low data with downsampling end


            discriminator.zero_grad()
            real_images = high_res.to(device)
            # real_images = high_res[0].to(device)

            # For generating Low res data sample
            # avg_pool = nn.AvgPool2d(kernel_size=Blur_size, stride=Blur_size)
            # downsampled_tensor = avg_pool(real_images)
            # # Upsample the downsampled tensor using bilinear interpolation
            # low_img = F.interpolate(downsampled_tensor, scale_factor=Blur_size, mode='bilinear')
            # fake_images = generator(low_img)
            # end

            # For the low resolution data already provided, Uncomment this
            # lr_img = low_res[0].to(device)
            lr_img = low_res.to(device)
            fake_images = generator(lr_img)


            real_labels = torch.ones(real_images.size(0)).to(device)
            fake_labels = torch.zeros(fake_images.size(0)).to(device) # .squeeze()
            real_output_non = discriminator(real_images)
            fake_output_non = discriminator(fake_images.detach())
            real_output = real_output_non.squeeze()
            fake_output = fake_output_non.squeeze()

            loss_d = criterion(real_output, real_labels) + criterion(fake_output, fake_labels)
            loss_d.backward()
            # Continued from the previous message:

            optimizer_d.step()

            # Train the generator
            generator.zero_grad()
            real_labels = torch.ones(fake_output.size(0)).to(device)
            fake_output = discriminator(fake_images).squeeze()
            loss_g = criterion(fake_output, real_labels) + l1_loss(fake_images, high_res[0].to(device))
            loss_g.backward()
            optimizer_g.step()

            # Print progress
            if i % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Discriminator Loss: {:.4f}, Generator Loss: {:.4f}"
                      .format(epoch+1, num_epochs, i+1, len(dataloader), loss_d.item(), loss_g.item()))

# Define hyperparameters and load the dataset
num_epochs = 10
batch_size = 32
learning_rate = 0.0002
l1_lambda = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = ImageFolder(root=dataroot, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# print(np.shape())# high_res
print(dataloader)
# Create the generator and discriminator networks, and define the loss functions and optimizers
generator = Generator().to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Train the GAN
train(generator, discriminator, dataloader, optimizer_g, optimizer_d, criterion)

# Generate a high-resolution image from a low-resolution input image
# img_pth = "path/to/low_res_image.jpg"
img_pth = "/content/1_copy.png"
low_res = Image.open(img_pth).convert("RGB")
low_res_tensor = transform(low_res).unsqueeze(0).to(device)
high_res_tensor = generator(low_res_tensor).squeeze(0).cpu()
high_res = transforms.ToPILImage()(high_res_tensor)
# high_res.save("path/to/high_res_image.jpg")
high_res.save("/content/high_res_image.jpg")