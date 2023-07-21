# mae and ganloss

import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import shutil
import pickle # for saving the model


def delete_output_folder():
    folder_name = "output_images"
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
        print(f"Folder '{folder_name}' and its contents have been deleted.")
    else:
        print(f"Folder '{folder_name}' does not exist.")


# Call the function to delete the folder
delete_output_folder()


def load_dataset():
    # Define paths
    dataset_folder = 'lol_dataset'
    low_folder = os.path.join(dataset_folder, 'low')
    high_folder = os.path.join(dataset_folder, 'high')

    # Sample loading
    transform = transforms.ToTensor()

    low_quality_img_paths = []
    high_quality_img_paths = []

    for filename in os.listdir(low_folder):
        low_qual_img_path = os.path.join(low_folder, filename)
        high_qual_img_path = os.path.join(high_folder, filename)

        # Store paths instead of loading images
        low_quality_img_paths.append(low_qual_img_path)
        high_quality_img_paths.append(high_qual_img_path)

    print(f"Loaded paths to {len(low_quality_img_paths)} image pairs successfully.")

    if len(low_quality_img_paths) != len(high_quality_img_paths):
        print("Error: Incomplete dataset.")
        return None, None

    return low_quality_img_paths, high_quality_img_paths


# Split dataset into train, validation, and test sets
def split_dataset(low_quality_imgs, high_quality_imgs):
    X = list(range(len(low_quality_imgs)))
    y = list(range(len(high_quality_imgs)))

    # Split into train (70%), validation (20%), test (10%)
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, test_size=1/3, random_state=42)

    # Create train, validation, and test datasets
    train_low_quality_imgs = [low_quality_imgs[i] for i in X_train]
    train_high_quality_imgs = [high_quality_imgs[i] for i in y_train]

    val_low_quality_imgs = [low_quality_imgs[i] for i in X_val]
    val_high_quality_imgs = [high_quality_imgs[i] for i in y_val]

    test_low_quality_imgs = [low_quality_imgs[i] for i in X_test]
    test_high_quality_imgs = [high_quality_imgs[i] for i in y_test]

    return (train_low_quality_imgs, train_high_quality_imgs,
            val_low_quality_imgs, val_high_quality_imgs,
            test_low_quality_imgs, test_high_quality_imgs)


num_images = 500  # DATASET INPUT AMOUNT


class LowLightDataset(Dataset):
    def __init__(self, low_quality_img_paths, high_quality_img_paths,
                 low_transform, high_transform):
        self.low_quality_img_paths = low_quality_img_paths
        self.high_quality_img_paths = high_quality_img_paths
        self.low_transform = low_transform
        self.high_transform = high_transform

    def __len__(self):
        return len(self.low_quality_img_paths)

    def __getitem__(self, idx):
        # Load images here
        low_img_path = self.low_quality_img_paths[idx]
        high_img_path = self.high_quality_img_paths[idx]

        # Open image file and apply transformations
        low_img = Image.open(low_img_path)
        high_img = Image.open(high_img_path)

        # Apply transformations
        low_img = self.low_transform(low_img)
        high_img = self.high_transform(high_img)

        return low_img, high_img


# Call the function to load the dataset
low_quality_imgs, high_quality_imgs = load_dataset()

# Split dataset into train(70%), validation(20%), and test(10%) sets
(train_low_quality_imgs, train_high_quality_imgs,
 val_low_quality_imgs, val_high_quality_imgs,
 test_low_quality_imgs, test_high_quality_imgs) = split_dataset(
    low_quality_imgs, high_quality_imgs)

# Print dataset statistics
print("Train dataset:")
print(f"Low-quality images: {len(train_low_quality_imgs)}")
print(f"High-quality images: {len(train_high_quality_imgs)}")

print("\nValidation dataset:")
print(f"Low-quality images: {len(val_low_quality_imgs)}")
print(f"High-quality images: {len(val_high_quality_imgs)}")

print("\nTest dataset:")
print(f"Low-quality images: {len(test_low_quality_imgs)}")
print(f"High-quality images: {len(test_high_quality_imgs)}")

# Define the transformations here, with augmentations
low_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=10),
])

high_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=10),
])

# Create DataLoader objects for train, validation, and test datasets
train_dataset = LowLightDataset(
    train_low_quality_imgs, train_high_quality_imgs, low_transform, high_transform)
val_dataset = LowLightDataset(
    val_low_quality_imgs, val_high_quality_imgs, low_transform, high_transform)
test_dataset = LowLightDataset(
    test_low_quality_imgs, test_high_quality_imgs, low_transform, high_transform)

import torch.nn as nn
import torch.nn.functional as F

# Generator (U-Net-like)
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Decoder
        self.tconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.tconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.tconv3 = nn.ConvTranspose2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))

        # Decoder with skip connections
        x = F.relu(self.tconv1(x3))
        x = F.relu(self.tconv2(x + x2))
        x = self.tconv3(x + x1)
        # Apply sigmoid to output
        x = torch.sigmoid(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.main = nn.Sequential(
            # Resize input to consistent size
            nn.AdaptiveAvgPool2d((64, 64)),
            # input is (in_channels) x 64 x 64
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (128) x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (256) x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (512) x 4 x 4
            nn.Conv2d(512, 3, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the directory for saving the images
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
print("DataLoaders created.")

# Models
generator = Generator().to(device)
discriminator = Discriminator().to(device)
print("Models loaded.")

# Loss function
beta = 0.01  # Weight for GAN loss
criterion_mae = nn.L1Loss()  # MAE loss
criterion_gan = nn.BCELoss()  # GAN loss


def loss_fn(real_outputs, fake_outputs, real_labels, fake_labels, beta):
    # MAE loss
    mae_loss = criterion_mae(real_outputs, fake_outputs)

    # Resize fake_outputs tensor to match the size of real_labels tensor
    fake_outputs_resized = F.interpolate(fake_outputs, size=real_labels.size()[2:], mode='bilinear')

    # GAN loss - calculate separately for each channel and then average
    gan_loss = criterion_gan(fake_outputs_resized, real_labels)
    gan_loss = gan_loss.mean()

    # Combined loss
    loss = mae_loss + beta * gan_loss

    return loss, mae_loss, gan_loss



print("Loss function loaded.")

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
print("Optimizers created.")

# Training loop
num_epochs = 20
d_losses = []
g_losses = []

# Plot losses
def plot_loss(d_losses, g_losses, epoch):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{output_dir}/loss_plot_{epoch}.png")  # Save the plot as an image
    plt.close()  # Close the plot to free up memory



# Define the accumulation steps
accumulation_steps = 2

import matplotlib.pyplot as plt

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch [{epoch}/{num_epochs}]")

    for i, (low_res, high_res) in enumerate(train_dataloader):
        print(f"Batch [{i+1}/{len(train_dataloader)}]")

        # Move data to device
        low_res = low_res.to(device)
        high_res = high_res.to(device)

        ##############################
        # Train the discriminator
        ##############################

        # Forward pass real samples through discriminator
        real_outputs = discriminator(high_res)
        real_labels = torch.ones_like(real_outputs).to(device)
        d_loss_real = criterion_gan(real_outputs, real_labels)

        # Generate fake images and forward pass through discriminator
        generated_imgs = generator(low_res)
        fake_outputs = discriminator(generated_imgs.detach())
        fake_labels = torch.zeros_like(fake_outputs).to(device)
        d_loss_fake = criterion_gan(fake_outputs, fake_labels)

        # Compute discriminator loss
        d_loss = d_loss_real + d_loss_fake

        # Normalize the discriminator loss
        d_loss = d_loss / accumulation_steps

        # Backpropagation
        d_loss.backward()

        # Optimization step and zero the gradients
        if (i+1) % accumulation_steps == 0:
            d_optimizer.step()
            d_optimizer.zero_grad()

        ##############################
        # Train the generator
        ##############################

        # Generate fake images and forward pass through discriminator
        generated_imgs = generator(low_res)
        fake_outputs = discriminator(generated_imgs)

        # Compute generator loss
        g_loss, mae_loss, gan_loss = loss_fn(high_res, generated_imgs, real_labels, fake_labels, beta)

        # Normalize the generator loss
        g_loss = g_loss / accumulation_steps

        # Backpropagation
        g_loss.backward()

        # Optimization step and zero the gradients
        if (i+1) % accumulation_steps == 0:
            g_optimizer.step()
            g_optimizer.zero_grad()

        ##############################
        # Print losses
        ##############################

        print(f"d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, mae_loss: {mae_loss.item()}, gan_loss: {gan_loss.item()}")

        # Save the losses
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

        ##############################
        # Save some generated samples
        ##############################

        if (i+1) % 10 == 0:
            save_image(generated_imgs, f"{output_dir}/generated_{epoch}_{i+1}.png")
            save_image(low_res, f"{output_dir}/input_{epoch}_{i+1}.png")
            save_image(high_res, f"{output_dir}/target_{epoch}_{i+1}.png")

    # Plot the losses
    plot_loss(d_losses, g_losses, epoch)

    ##############################
    # Validation after each epoch
    ##############################
    generator.eval()  # Set the generator to evaluation mode
    with torch.no_grad():  # Disable gradient calculations
        val_losses = []  # List to store validation losses
        for low_res, high_res in val_dataloader:
            # Move data to device
            low_res = low_res.to(device)
            high_res = high_res.to(device)

            # Generate high-resolution images from low-resolution inputs
            generated_imgs = generator(low_res)

            # Compute loss
            val_loss = criterion_mae(high_res, generated_imgs)

            # Save the loss
            val_losses.append(val_loss.item())

        # Compute average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Validation Loss: {avg_val_loss}")

    generator.train()  # Set the generator back to training mode

# Testing after training...
generator.eval()  # Set the generator to evaluation mode
with torch.no_grad():  # Disable gradient calculations
    test_losses = []  # List to store test losses
    for i, (low_res, high_res) in enumerate(test_dataloader):
        # Move data to device
        low_res = low_res.to(device)
        high_res = high_res.to(device)

        # Generate high-resolution images from low-resolution inputs
        generated_imgs = generator(low_res)

        # Compute loss
        test_loss = criterion_mae(high_res, generated_imgs)

        # Save the loss
        test_losses.append(test_loss.item())

        # Save some generated samples
        if i % 10 == 0:
            save_image(generated_imgs, f"{output_dir}/test_generated_{i}.png")
            save_image(low_res, f"{output_dir}/test_input_{i}.png")
            save_image(high_res, f"{output_dir}/test_target_{i}.png")

    # Compute average test loss
    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f"Test Loss: {avg_test_loss}")

# Save the trained model
model_path = "models/trained_lowlight_rgb_fullres.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Create the directory if it doesn't exist
with open(model_path, "wb") as file:
    pickle.dump(generator, file)
print(f"Trained model saved at '{model_path}'")


