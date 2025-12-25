import torch
from pytorch_quantization import quant_modules
import os
from torchvision import models, transforms
from PIL import Image
import argparse
from tqdm import tqdm


# Create an argument parser
parser = argparse.ArgumentParser(description="Image classification using MobileNetV2")

# Add an argument for the image path
parser.add_argument("image_path", type=str, help="Path to the input image")
parser.add_argument("--labels", type=str, help="Choose a labels")
# Parse the command line arguments
args = parser.parse_args()

# Use the provided image path
img_path = args.image_path
labels_path = args.labels

# Search for CUDA device
cuda = torch.device('cuda')

# Load pretrained MobileNetV2 model and send it to CUDA device
quant_modules.initialize()
model = models.mobilenet_v2(pretrained=True)
model.eval()
model.cuda()

# Define preprocess option for input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )])

image_paths = [os.path.join(img_path, filename) for filename in os.listdir(img_path) if filename.endswith(('.jpg', '.jpeg', '.png'))]
images = [Image.open(image_path) for image_path in image_paths]

# Preprocess images and send them to CUDA device in a batch
batch = torch.stack([transform(image).to(device=cuda) for image in images])


# Set neural network model as evaluation mode(= inference mode) 


# Inference
out = model(batch)

# Load class labels
with open(labels_path) as f:
    labels = [line.strip().split(',')[1][1:] for line in f.readlines()]

# Find maximum output index
_, index = torch.max(out, 1)

# Softmax
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

# Final result
print(labels[index[0]], percentage[index[0]].item())
