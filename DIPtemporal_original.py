import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

class DIPNetwork(nn.Module):
    def __init__(self, input_channels=3, noise_std=0.2):
        super(DIPNetwork, self).__init__()
        self.noise_std = noise_std
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, noise=None):
        if self.training:
            if noise is None:
                noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        x = self.encoder(x)
        x = self.decoder(x)
        return x, noise

import cv2
import os
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

def load_frames_from_folder(folder_path, frame_size=(256, 256)):
    frames = []
    for filename in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, filename)
        frame = cv2.imread(path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)
    return frames

def load_frames_from_video(video_path, frame_size=(256, 256)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)
    cap.release()
    return frames

class FramesDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames
        self.transform = transforms.Compose([
            ToTensor(),
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        frame = self.transform(frame)
        return frame

def process_frames_in_batches(frames, batch_size, num_iterations=500, device='cuda'):
    dataset = FramesDataset(frames)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    processed_frames = []
    previous_noise = None

    for batch in dataloader:
        batch = batch.to(device)
        dip_network = DIPNetwork().to(device)
        optimizer = optim.Adam(dip_network.parameters(), lr=0.0001)

        # Check if previous_noise exists and matches the current batch size
        if previous_noise is not None and previous_noise.size(0) == batch.size(0):
            current_noise = previous_noise
        else:
            current_noise = None  # Reset or create new noise for mismatched sizes

        for iteration in range(num_iterations):
            optimizer.zero_grad()
            output, current_noise = dip_network(batch, current_noise)
            loss = torch.mean((output - batch) ** 2)
            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:  # Adjust this value to control how often you want to see updates
              print(f"Iteration {iteration}/{num_iterations}, Loss: {loss.item()}")

        processed_batch = output.detach().cpu().numpy()
        previous_noise = current_noise.detach()  # Detach and propagate the noise to the next batch

        for i in range(processed_batch.shape[0]):
            frame = processed_batch[i].transpose(1, 2, 0)
            frame = (frame * 255).clip(0, 255).astype('uint8')
            processed_frames.append(frame)

    return processed_frames

class FramesDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames
        self.transform = transforms.Compose([
            ToTensor(),
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        frame = self.transform(frame)
        return frame

def save_frames_to_video(frames, output_path, fps):
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

import struct
from PIL import Image
import io
import os

def read_dat_images(file_path, image_width, image_height, save_path):
    # Open the .dat file in binary mode
    with open(file_path, 'rb') as file:
        image_index = 1
        while True:
            # Read raw image data based on known size (for grayscale images)
            image_data = file.read(image_width * image_height * 3)  # Adjust this line if your images are color (e.g., *3 for RGB)
            if not image_data:
                break  # Exit loop if no more data

            # Convert the raw image data to an image
            # Create an image object using PIL with the appropriate mode ('L' for grayscale, 'RGB' for color)
            image = Image.frombytes('RGB', (image_width, image_height), image_data)

            # Save the image as a PNG file
            image.save(os.path.join(save_path, f'image_{image_index}.png'))
            print(f'Saved image_{image_index}.png')
            image_index += 1

    print("All images extracted and saved.")

# Specify the path to your .dat file and the dimensions of the images
read_dat_images('so211.dat', 256, 256, '/content/processed_images')  # Adjust image dimensions as necessary

frames = load_frames_from_folder('/content/processed_images')

# Process frames (adjust batch_size and num_iterations as needed)
processed_frames = process_frames_in_batches(frames, batch_size=6, num_iterations=5000)

# Use original_fps when saving the output video
save_frames_to_video(processed_frames, 'output_video.mp4', fps=18)