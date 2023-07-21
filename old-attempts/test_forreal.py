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


def delete_output_folder():
    folder_name = "output_images"
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
        print(f"Folder '{folder_name}' and its contents have been deleted.")
    else:
        print(f"Folder '{folder_name}' does not exist.")

# Call the function to delete the folder
delete_output_folder()

def convert_folder_to_Y(input_folder, output_folder):
    os.makedirs(os.path.join(output_folder, 'Y'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'Cr'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'Cb'), exist_ok=True)

    for filename in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, filename)

        if os.path.isfile(input_image_path):
            with Image.open(input_image_path) as image:
                # Convert image to YCrCb
                ycrcb_image = image.convert('YCbCr')

                # Split YCrCb image into individual channels
                y_channel, cr_channel, cb_channel = ycrcb_image.split()

                # Save the Y channel as a separate grayscale image
                output_y_image_path = os.path.join(output_folder, 'Y',  filename)
                y_channel.save(output_y_image_path, format='JPEG')

                # Save the Cr channel
                output_cr_image_path = os.path.join(output_folder, 'Cr', filename)
                cr_channel.save(output_cr_image_path, format='JPEG')

                # Save the Cb channel
                output_cb_image_path = os.path.join(output_folder, 'Cb',  filename)
                cb_channel.save(output_cb_image_path, format='JPEG')



convert_folder_to_Y("lol_dataset/low", "lol_converted_dataset/low")
convert_folder_to_Y("lol_dataset/high", "lol_converted_dataset/high")

def load_dataset():
    # Define paths
    dataset_folder = 'lol_converted_dataset'
    low_folder = os.path.join(dataset_folder, 'low', 'Y')
    high_folder = os.path.join(dataset_folder, 'high', 'Y')


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
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=1/3, random_state=42)

    # Create train, validation, and test datasets
    train_low_quality_imgs = [low_quality_imgs[i] for i in X_train]
    train_high_quality_imgs = [high_quality_imgs[i] for i in y_train]

    val_low_quality_imgs = [low_quality_imgs[i] for i in X_val]
    val_high_quality_imgs = [high_quality_imgs[i] for i in y_val]

    test_low_quality_imgs = [low_quality_imgs[i] for i in X_test]
    test_high_quality_imgs = [high_quality_imgs[i] for i in y_test]

    return train_low_quality_imgs, train_high_quality_imgs, val_low_quality_imgs, val_high_quality_imgs, test_low_quality_imgs, test_high_quality_imgs

num_images = 500 # DATASET INPUT AMOUNT

class LowLightDataset(Dataset):
    def __init__(self, low_quality_img_paths, high_quality_img_paths, low_transform, high_transform):
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
train_low_quality_imgs, train_high_quality_imgs, val_low_quality_imgs, val_high_quality_imgs, test_low_quality_imgs, test_high_quality_imgs = split_dataset(low_quality_imgs, high_quality_imgs)

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

# Define the transformations here
low_transform = transforms.Compose([
    transforms.Resize((100, 150)),  # Resize for low-quality images
    transforms.ToTensor(),
])

high_transform = transforms.ToTensor()  # Only convert to tensor for high-quality images

# Create DataLoader objects for train, validation, and test datasets
train_dataset = LowLightDataset(train_low_quality_imgs, train_high_quality_imgs, low_transform, high_transform)
val_dataset = LowLightDataset(val_low_quality_imgs, val_high_quality_imgs, low_transform, high_transform)
test_dataset = LowLightDataset(test_low_quality_imgs, test_high_quality_imgs, low_transform, high_transform)

# import torch
import torch.nn as nn
import torch.nn.functional as F

# Generator (U-Net-like)
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Decoder
        self.tconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.tconv4 = nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))

        # Decoder with skip connections
        x = F.relu(self.tconv1(x3))
        if x2.size() != x.size():
            x = F.interpolate(x, size=(x2.size(2), x2.size(3)))
        x = F.relu(self.tconv2(x + x2))
        if x1.size() != x.size():
            x = F.interpolate(x, size=(x1.size(2), x1.size(3)))
        x = F.relu(self.tconv3(x + x1))
        x = self.tconv4(x)

        return torch.tanh(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels = 1):
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
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
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
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print("DataLoaders created.")

# Models
generator = Generator().to(device)
discriminator = Discriminator().to(device)
for param in discriminator.parameters():
    param.requires_grad = True
print("Models loaded.")
beta = 0.1
# Losses
# Manual implementation of LSGAN loss
def lsgan_loss(fakes, reals):
  loss_fake = 0.5 * torch.mean((fakes - 0)**2)  
  loss_real = 0.5 * torch.mean((reals - 1)**2)
  return loss_fake + loss_real
# Loss functions
mae_criterion = nn.L1Loss()
def generator_loss(fakes, reals):
  mae = mae_criterion(fakes, reals)
  gan = lsgan_loss(fakes, reals)
  return mae, gan



# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=0.00001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.00001)
print("Optimizers created.")

# Training loop
num_epochs = 100
d_losses = []
g_losses = []

# Plot losses
def plot_loss(d_losses, g_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Define the accumulation steps
accumulation_steps = 2

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
        d_loss_real = generator_loss(real_outputs, real_labels)

        # Generate fake images and forward pass through discriminator
        generated_imgs = generator(low_res)
        fake_outputs = discriminator(generated_imgs.detach())
        fake_labels = torch.zeros_like(fake_outputs).to(device)
        d_loss_fake = generator_loss(fake_outputs, fake_labels)

        # Compute discriminator loss
        #d_loss = d_loss_real + d_loss_fake
        _, d_loss_real_tensor = d_loss_real
        _, d_loss_fake_tensor = d_loss_fake
        real_loss = torch.sum(d_loss_real_tensor)
        fake_loss = torch.sum(d_loss_fake_tensor)
        d_loss = real_loss + fake_loss
        print("d_loss: ", d_loss)
        # Detach before normalize, and normalize the discriminator loss
        d_loss = d_loss.detach() / accumulation_steps

        # Backpropagation
        d_loss.backward()
        
        # Optimization step and zero the gradients
        if (i+1) % accumulation_steps == 0:
            d_optimizer.step()
            d_optimizer.zero_grad()

        ##############################
        # Train the generator
        ##############################

        # Forward pass fake images through discriminator
        fake_outputs = discriminator(generated_imgs)
        g_loss = generator_loss(fake_outputs, real_labels)


        # Detach before normalizing and, normalize the generator loss
        g_loss = g_loss.detach() / accumulation_steps

        # Backpropagation
        g_loss.backward()

        # Optimization step and zero the gradients
        if (i+1) % accumulation_steps == 0:
            g_optimizer.step()
            g_optimizer.zero_grad()

        # Save losses for plotting
        d_losses.append(d_loss.item() * accumulation_steps)  # Multiply by accumulation_steps to get the actual loss value
        g_losses.append(g_loss.item() * accumulation_steps)  # Multiply by accumulation_steps to get the actual loss value

        print(f"Batch [{i+1}/{len(train_dataloader)}], Generator Loss: {g_loss.item() * accumulation_steps}, Discriminator Loss: {d_loss.item() * accumulation_steps}")

        # Save the generated images every 10 batches
        if (i+1) % 10 == 0:
            for j, img in enumerate(generated_imgs):
                save_image(0.5 * img + 0.5, os.path.join(output_dir, f"generated_img_epoch{epoch}_batch{i+1}_img{j}.png"))


    # Save the generated images at the end of each epoch
    save_image(generated_imgs, os.path.join(output_dir, f"generated_img_epoch{epoch}.png"))

# Plot losses
plot_loss(d_losses, g_losses)



