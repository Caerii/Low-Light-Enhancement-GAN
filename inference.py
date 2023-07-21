import torch
import pickle
from PIL import Image
from torchvision import transforms

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

# Load the trained model
model_path = "models/trained_lowlight_rgb_fullres.pkl"
with open(model_path, "rb") as file:
    generator = pickle.load(file)
generator.eval()  # Set the model to evaluation mode

# Load the input low-lit image
input_image_path = "pickle.jpg"
input_image = Image.open(input_image_path)

# Resize the input image to 400x600
resized_image = input_image.resize((600, 400))

# Apply necessary transformations to the resized image
transform = transforms.ToTensor()
input_tensor = transform(resized_image).unsqueeze(0)

# Move the input tensor to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)

# Generate the enhanced image
with torch.no_grad():
    enhanced_tensor = generator(input_tensor)

# Move the enhanced tensor back to the CPU and convert it to a PIL image
enhanced_image = enhanced_tensor.squeeze(0).cpu()
enhanced_image = transforms.ToPILImage()(enhanced_image)

# Save the enhanced image
output_image_path = "output_enhanced_image.jpg"
enhanced_image.save(output_image_path)
print("Enhanced image saved to {}".format(output_image_path))
