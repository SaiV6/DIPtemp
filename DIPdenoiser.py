from PIL import Image
import numpy as np

def add_noise_to_image(image_path, output_path, noise_level=100):
    """
    Add more noise to an image.

    Parameters:
    - image_path: Path to the input image.
    - output_path: Path where the noisy image will be saved.
    - noise_level: The standard deviation of the Gaussian noise to be added. Default is 50.
    """
    # Open the image
    image = Image.open(image_path).convert('RGB')

    # Convert image to numpy array
    image_array = np.asarray(image)

    # Generate Gaussian noise with increased standard deviation
    noise = np.random.normal(0, noise_level, image_array.shape)

    # Add the noise to the image
    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)

    # Convert back to an image
    noisy_image = Image.fromarray(noisy_image)

    # Save the noisy image
    noisy_image.save(output_path)


# Example usage
image_path = '/content/Unetcopy.tiff'  # Path to your image
output_path = '/content/test_noise.png'  # Path to save the noisy image
add_noise_to_image(image_path, output_path)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import matplotlib.pyplot as plt

class DIPCNN(nn.Module):
    def __init__(self):
        super(DIPCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Remove Upsample or replace with appropriate layers
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)



# Function to load an image and convert it to a tensor
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return to_tensor(image).unsqueeze(0) # Add batch dimension

# Function to train the model using DIP principles
def train_DIP_model(model, noisy_image_tensor, num_epochs=1000, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(noisy_image_tensor)
        loss = torch.mean((output - noisy_image_tensor)**2)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    return model

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load your noisy image and convert to tensor
noisy_image_path = output_path
noisy_image_tensor = load_image(noisy_image_path).to(device)

# Initialize the model
model = DIPCNN().to(device)

# Train the model
model = train_DIP_model(model, noisy_image_tensor, num_epochs=1000, lr=0.001)

# Generate the denoised image
with torch.no_grad():
    denoised_image_tensor = model(noisy_image_tensor)
denoised_image = to_pil_image(denoised_image_tensor.squeeze(0).cpu())

# Display the original noisy and denoised images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Noisy Image')
plt.imshow(load_image(noisy_image_path).squeeze(0).permute(1, 2, 0))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Denoised Image')
plt.imshow(denoised_image)
plt.axis('off')
plt.show()
