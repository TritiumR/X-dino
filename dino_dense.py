import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import pandas as pd
import numpy as np
import argparse
import imageio
import cv2
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image
import math
from torchvision import transforms
from transformers.image_utils import load_image
import torch.nn.functional as F


def make_transform(resize_size: int = 720):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])
    # return transforms.Compose([to_tensor, normalize])

parser = argparse.ArgumentParser(description="Test DINOv3")
parser.add_argument("--file_path", type=str, help="Path to the image file")
parser.add_argument("--folder_path", type=str, help="Path to the image folder")
args = parser.parse_args()

# Load DINOv3 model and processor directly
print("Loading DINOv3 model...")
# model_name = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name, 
)

transform = make_transform()

def compute_patch_cosine_similarity(features, features_2, pixel_position, original_size):
    """
    Create a heatmap visualization of DINOv3 features using the same patch-based approach
    """
    # Convert to tensor if needed
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features)
    
    # Remove batch dimension
    features_no_batch = features.squeeze(0)  # -> (201, 1024)

    if not isinstance(features_2, torch.Tensor):
        features_2 = torch.tensor(features_2)
    
    # Remove batch dimension
    features_2_no_batch = features_2.squeeze(0)  # -> (201, 1024)
    
    num_tokens = features_no_batch.shape[0]
    num_patches_guess = int(math.sqrt(num_tokens))**2

    # h = w = int(math.sqrt(num_patches_guess))
    
    num_special_tokens = num_tokens - num_patches_guess
    
    # Extract patch features (exclude special tokens like [CLS])
    patch_features = features_no_batch[num_special_tokens:, :]

    patch_features = patch_features.reshape(14, 14, -1)

    print(patch_features.shape)

    h, w = original_size

    # Reshape to (1, C, H, W) format for interpolation
    patch_features = patch_features.permute(2, 0, 1).unsqueeze(0)  # (1, C, 14, 14)
    upsample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=False)
    patch_features = upsample(patch_features).squeeze(0)  # (C, h, w)
    patch_features = patch_features.permute(1, 2, 0)  # (h, w, C)

    pixel_features = patch_features[pixel_position[0], pixel_position[1], :]
    
    # Compute cosine similarity between the selected patch feature and all patch features from the second image
    # Normalize features for cosine similarity
    pixel_features_norm = F.normalize(pixel_features.unsqueeze(0), p=2, dim=1)  # (1, 1024)
    
    # Extract patch features from the second image (exclude special tokens)
    patch_features_2 = features_2_no_batch[num_special_tokens:, :]  # (num_patches, 1024)
    patch_features_2 = patch_features_2.reshape(14, 14, -1)

    # Reshape to (1, C, H, W) format for interpolation
    patch_features_2 = patch_features_2.permute(2, 0, 1).unsqueeze(0)  # (1, C, 14, 14)
    upsample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=False)
    patch_features_2 = upsample(patch_features_2).squeeze(0)  # (C, h, w)
    patch_features_2 = patch_features_2.permute(1, 2, 0)  # (h, w, C)

    # Reshape patch_features_2 for normalization: (h, w, C) -> (h*w, C)
    patch_features_2_flat = patch_features_2.reshape(-1, patch_features_2.shape[-1])  # (h*w, C)
    patch_features_2_norm = F.normalize(patch_features_2_flat, p=2, dim=1)  # (h*w, C)
    
    # Compute cosine similarity: (1, C) @ (C, h*w) = (1, h*w)
    similarity = torch.mm(pixel_features_norm, patch_features_2_norm.t()).squeeze(0)  # (h*w,)
    
    # Reshape similarity to match the patch grid
    similarity = similarity.reshape(h, w)
    
    return similarity


print("Processing frames and extracting features...")

output_folder = os.path.join("output", args.file_path.split("/")[-1].split(".")[0])
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image = load_image(args.file_path)

inputs = processor(images=image, return_tensors="pt").to(model.device)
with torch.inference_mode():
    outputs = model(**inputs)

# print(outputs.keys())
features = outputs.last_hidden_state

vis_image = np.array(image).astype(np.uint8)

h, w = vis_image.shape[:2]

point_file = args.file_path.replace(".png", ".pkl")

points = np.array(pickle.load(open(point_file, "rb"))).astype(np.int32)

points = points[:, 1:]

print(points)

for point_id, point in enumerate(points):
    u = point[0]
    v = point[1]
    vis_image = cv2.circle(vis_image, (u, v), 5, (point_id * 255 // len(points), 255 - point_id * 255 // len(points), 0), -1)

# save the images
vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(output_folder, "reference.png"), vis_image)

for file_name in os.listdir(args.folder_path):
    if not file_name.endswith(".png"):
        continue

    file_path = os.path.join(args.folder_path, file_name)
    image_2 = load_image(file_path)

    inputs_2 = processor(images=image_2, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs_2 = model(**inputs_2)
    features_2 = outputs_2.last_hidden_state

    print(features_2.shape)

    vis_image_2 = np.array(image_2).astype(np.uint8)

    # find the most similar patch
    for point_id, point in enumerate(points):
        u = point[0]
        v = point[1]
        similarity = compute_patch_cosine_similarity(features, features_2, (v, u), (h, w))

        print(similarity.shape)

        most_similar_patch = np.argmax(similarity)

        print(most_similar_patch)

        # Convert flattened index back to 2D coordinates
        u = int(most_similar_patch % w)  # x-coordinate (column)
        v = int(most_similar_patch // w)  # y-coordinate (row)

        print(u, v)

        vis_image_2 = cv2.circle(vis_image_2, (u, v), 5, (point_id * 255 // len(points), 255 - point_id * 255 // len(points), 0), -1)

    vis_image_2 = cv2.cvtColor(vis_image_2, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_folder, file_name.replace(".png", "_corr.png")), vis_image_2)

