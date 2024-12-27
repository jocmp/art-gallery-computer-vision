import torch
from torchvision import models
from PIL import Image
import os

import torchvision.transforms as transforms

# Load the model
model = torch.load('deepaugment_and_augmix.pth.tar')
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to classify an image
def classify_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Directory containing images
image_dir = 'images'

# Classify each image in the directory
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    if os.path.isfile(image_path):
        classification = classify_image(image_path)
        print(f'Image: {image_name}, Classification: {classification}')
