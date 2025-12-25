import torch
from torchvision import models, transforms, datasets
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description="Image classification using MobileNetV2")

# Add an argument for the batch size
parser.add_argument("--imagenet_path", type=str, help="ImageNet path")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
parser.add_argument("--iter_size", type=int, default=4, help="iter size for inference")
parser.add_argument("--labels", type=str, help="Choose a labels")

# Parse the command line arguments
args = parser.parse_args()

img_path = args.imagenet_path

# Search for CUDA device
cuda = torch.device('cuda')

# Load pretrained MobileNetV2 model and send it to CUDA device
model = models.mobilenet_v2(pretrained=True)
model.to(device=cuda)

# Define preprocess option for input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )])

# Load ImageNet dataset with the desired batch size
imagenet_data = datasets.ImageNet(root=img_path, split="val", transform=transform)
data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=True)

# Set neural network model as evaluation mode(= inference mode) 
model.eval()

# Counter to track the number of batches processed
iter = 0
iter_max = args.iter_size

for batch in data_loader:
    batch = batch.to(device=cuda)

    # Inference
    out = model(batch)

    # Load class labels
    with open(args.labels) as f:
        labels = [line.strip().split(',')[1][1:] for line in f.readlines()]

    # Find maximum output indices for each batch
    _, indices = torch.max(out, 1)

    # Softmax probabilities for each batch
    percentages = torch.nn.functional.softmax(out, dim=1)

    for i in range(args.batch_size):
        print(labels[indices[i]], percentages[i, indices[i]].item())

    # Increment the batch count
    iter += 1

    # Stop after processing 4 batches (adjust this number as needed)
    if iter == iter_max:
        break
