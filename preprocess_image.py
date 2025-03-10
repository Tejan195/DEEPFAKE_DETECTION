import cv2
import os
import numpy as np
import torch
from torchvision import transforms

def preprocess_image(image_path, image_size=(299, 299)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, image_size)  # Resize to model input size

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    return transform(image).unsqueeze(0)  # Add batch dimension

# Example Usage
image_path = r"C:\Users\tejan\OneDrive\Desktop\Deefake_Detection_Model\test_frames\frame_0.jpg"
processed_image = preprocess_image(image_path)
