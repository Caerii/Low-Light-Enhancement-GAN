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
import matplotlib.pyplot as plt
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
#### METRICS ####
from torchvision.transforms.functional import to_pil_image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(output, target):
    output = output.clamp(0, 1)  # Clamp values between 0 and 1
    mse = torch.mean((output - target) ** 2)  # Calculate mean squared error
    psnr = 10 * torch.log10(1 / mse)  # Calculate PSNR using MSE
    return psnr

import numpy as np
from skimage.metrics import structural_similarity as ssim_metric
from torchvision.transforms.functional import to_pil_image

def calculate_ssim(output, target):
    ssim_values = []

    for i in range(output.size(0)):
        # Convert the tensor to NumPy array and scale values to [0, 255] for RGB images
        output_np = np.array(to_pil_image((output[i].cpu() * 255).byte()))
        target_np = np.array(to_pil_image((target[i].cpu() * 255).byte()))

        # Determine the smaller side of the images
        smaller_side = min(output_np.shape[:2])

        # Set win_size to an odd value less than or equal to the smaller side of the images
        win_size = min(smaller_side, 3) if smaller_side % 2 == 1 else min(smaller_side - 1, 3)


        # Compute the structural similarity index
        ssim_val = ssim_metric(output_np, target_np, win_size=win_size, multichannel=True)
        ssim_values.append(ssim_val)

    mean_ssim = torch.tensor(ssim_values).mean()  # Calculate mean SSIM across the batch
    return mean_ssim

def calculate_mae(output, target):
    mae = torch.mean(torch.abs(output - target))  # Calculate mean absolute error
    return mae


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

class FIDEvaluationDataset(Dataset):
    def __init__(self, low_quality_img_paths, high_quality_img_paths):
        self.low_quality_img_paths = low_quality_img_paths
        self.high_quality_img_paths = high_quality_img_paths
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.low_quality_img_paths)

    def __getitem__(self, idx):
        # Load images here
        low_img_path = self.low_quality_img_paths[idx]
        high_img_path = self.high_quality_img_paths[idx]

        # Open image file
        low_img = Image.open(low_img_path)
        high_img = Image.open(high_img_path)

        # Convert images to tensors
        low_img = self.transform(low_img)
        high_img = self.transform(high_img)

        return low_img, high_img
    
from torchvision.transforms import Resize, CenterCrop, Normalize, Compose

def fid_preprocess(image):
    transform = Compose([
        Resize((299, 299)),            # Resize to 299x299 pixels
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ])
    return transform(image)


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
    
])

high_transform = transforms.Compose([
    transforms.ToTensor(),
   
])

# Create DataLoader objects for train, validation, and test datasets
train_dataset = LowLightDataset(
    train_low_quality_imgs, train_high_quality_imgs, low_transform, high_transform)
val_dataset = LowLightDataset(
    val_low_quality_imgs, val_high_quality_imgs, low_transform, high_transform)
test_dataset = LowLightDataset(
    test_low_quality_imgs, test_high_quality_imgs, low_transform, high_transform)
# Dataset for FID evaluation (Separate from GAN data)
fid_eval_dataset = FIDEvaluationDataset(val_low_quality_imgs, val_high_quality_imgs)

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

batch_size = 4

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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

# Calculate the Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Mean Absolute Error (MAE)
def save_metrics_plot(psnr_values, ssim_values, mae_values, epoch):
    plt.figure(figsize=(10, 5))
    plt.title("Metrics over Time")
    plt.plot(psnr_values, label="PSNR")
    plt.plot(ssim_values, label="SSIM")
    plt.plot(mae_values, label="MAE")
    plt.xlabel("Iterations")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.savefig(f"{output_dir}/metrics_plot_{epoch}.png")
    plt.close()  # Close the plot to free up memory

from scipy import linalg

# Calculate the Frechet Inception Distance (FID) score
def calculate_fid(real_features, generated_features):
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_generated, sigma_generated = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

    # Calculate the squared sum of differences between means
    ssdiff = np.sum((mu_real - mu_generated) ** 2.0)

    # Calculate the product of covariance matrices
    covmean = linalg.sqrtm(sigma_real.dot(sigma_generated))

    # Check for imaginary numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate the FID score
    fid = ssdiff + np.trace(sigma_real + sigma_generated - 2.0 * covmean)

    return fid


# Optimizers, purpose is to update the weights
g_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
print("Optimizers created.")

# Training loop
num_epochs = 15
d_losses = []
g_losses = []
psnr_values = []
ssim_values = []
mae_values = []

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

# Csv file to store the metrics
import csv
metrics_data = []
loss_data = []

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
        # Calculate metrics
        ##############################

        psnr = calculate_psnr(generated_imgs, high_res)
        ssim = calculate_ssim(generated_imgs, high_res)
        mae = calculate_mae(generated_imgs, high_res)

        # Save metrics values
        psnr_values.append(psnr.item())
        ssim_values.append(ssim.item())
        mae_values.append(mae.item())

        # Append metrics data to the list
        metrics_data.append([psnr.item(), ssim.item(), mae.item()])

        # Append loss data to the list
        loss_data.append([d_loss.item() * accumulation_steps, g_loss.item() * accumulation_steps, mae_loss.item(), gan_loss.item()])

        ##############################
        # Print metrics and losses
        ##############################

        print(f"Batch [{i+1}/{len(train_dataloader)}], "
              f"Generator Loss: {g_loss.item() * accumulation_steps}, "
              f"Discriminator Loss: {d_loss.item() * accumulation_steps}, "
              f"PSNR: {psnr}, SSIM: {ssim}, MAE: {mae}")
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

    # Save metrics plot
    save_metrics_plot(psnr_values, ssim_values, mae_values, epoch)
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

import matplotlib.pyplot as plt
from torchvision.models import inception_v3

# Dataloader for FID evaluation (No shuffling)
fid_eval_dataloader = DataLoader(fid_eval_dataset, batch_size=batch_size, shuffle=False)


# Pre-trained Inception model
inception_model = inception_v3(pretrained=True, transform_input=False)
inception_model.eval()
inception_model.to(device)

fid_real_features_list = []
fid_generated_features_list = []

# FID evaluation loop
for i, (low_res, high_res) in enumerate(fid_eval_dataloader):
    # Move data to device (if required)
    low_res = low_res.to(device)
    high_res = high_res.to(device)

    # Generate fake images and forward pass through discriminator (if using GAN)
    generated_imgs = generator(low_res)

    # Extract features from real and generated images
    with torch.no_grad():
        real_features = inception_model(fid_preprocess(high_res)).detach().cpu().numpy()
        generated_features = inception_model(fid_preprocess(generated_imgs)).detach().cpu().numpy()

        fid_real_features_list.append(real_features)
        fid_generated_features_list.append(generated_features)

# Concatenate the features for all batches
fid_real_features_all = np.concatenate(fid_real_features_list)
fid_generated_features_all = np.concatenate(fid_generated_features_list)

# Calculate the FID score
fid_score = calculate_fid(fid_real_features_all, fid_generated_features_all)

# Save FID to CSV file (similar to previous step)
fid_csv_filename = "fid_scores.csv"

with open(fid_csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["FID Score"])  # Write header
    writer.writerow([fid_score])  # Write FID score to the file

print(f"FID score exported to '{fid_csv_filename}'")


# Save the trained model
model_path = "models/trained_lowlight_metrics.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Create the directory if it doesn't exist
with open(model_path, "wb") as file:
    pickle.dump(generator, file)
print(f"Trained model saved at '{model_path}'")

# Plot metrics over time
save_metrics_plot(psnr_values, ssim_values, mae_values, num_epochs)

# Write metrics to a CSV file
csv_filename = "metrics_data.csv"

with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["PSNR", "SSIM", "MAE"])  # Write header
    writer.writerows(metrics_data)  # Write metrics data to the file

print(f"Metrics data exported to '{csv_filename}'")

# Write loss to a CSV file
loss_csv_filename = "loss_data.csv"

with open(loss_csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Discriminator Loss", "Generator Loss", "MAE Loss", "GAN Loss"])  # Write header
    writer.writerows(loss_data)  # Write loss data to the file

print(f"Loss data exported to '{loss_csv_filename}'")


########################

from PIL import Image

# Function to create an animated GIF from the saved images
def create_animation(output_dir, output_file, fps):
    images = []
    filenames = sorted(os.listdir(output_dir))
    for filename in filenames:
        if filename.startswith("generated_"):
            image_path = os.path.join(output_dir, filename)
            images.append(Image.open(image_path))
    duration = int(1000 / fps)  # Duration in milliseconds
    images[0].save(output_file, save_all=True, append_images=images[1:], duration=duration, loop=0)

# Create the animation
animation_output_file = os.path.join(output_dir, "generated_animation.gif")
create_animation(output_dir, animation_output_file, fps=5)  # Adjust the fps as needed
print(f"Animation saved at '{animation_output_file}'")
