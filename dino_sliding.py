import torch
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

def extract_sliding_window_features(image, model, processor, patch_size_ratio=0.2):
    """
    Extract features using sliding window approach.
    Takes 1/5 of the image dimensions and creates overlapping patches.
    Returns dense features of shape (70, 70, feature_dim).
    """
    # Get image dimensions
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Calculate patch size (1/5 of image dimensions)
    patch_h = int(h * patch_size_ratio)
    patch_w = int(w * patch_size_ratio)
    
    # Calculate step size for sliding window
    # We want 70x70 output, so step size should be calculated accordingly
    step_h = max(1, h // 5)
    step_w = max(1, w // 5)
    
    # Calculate number of patches
    num_patches_h = (h - patch_h) // step_h + 1
    num_patches_w = (w - patch_w) // step_w + 1
    
    # Ensure we have exactly 70x70 patches
    if num_patches_h > 5:
        step_h = (h - patch_h) // (5 - 1)
        num_patches_h = 5
    if num_patches_w > 5:
        step_w = (w - patch_w) // (5 - 1)
        num_patches_w = 5
    
    # Initialize feature tensor
    all_features = []
    
    for i in range(num_patches_h):
        row_features = []
        for j in range(num_patches_w):
            # Calculate patch coordinates
            start_h = i * step_h
            start_w = j * step_w
            
            # Ensure patch doesn't exceed image boundaries
            end_h = min(start_h + patch_h, h)
            end_w = min(start_w + patch_w, w)
            
            # Adjust start coordinates if patch would exceed boundaries
            if end_h - start_h < patch_h:
                start_h = end_h - patch_h
            if end_w - start_w < patch_w:
                start_w = end_w - patch_w
            
            # Extract patch
            patch = image.crop((start_w, start_h, end_w, end_h))
            
            # Process patch
            inputs = processor(images=patch, return_tensors="pt").to(model.device)
            
            with torch.inference_mode():
                outputs = model(**inputs)
                patch_features = outputs.last_hidden_state
            
            # Extract patch features (exclude special tokens)
            patch_features_no_batch = patch_features.squeeze(0)  # Remove batch dimension
            num_tokens = patch_features_no_batch.shape[0]
            num_patches_guess = int(math.sqrt(num_tokens))**2
            num_special_tokens = num_tokens - num_patches_guess
            
            # Extract patch features (exclude special tokens like [CLS])
            patch_features_only = patch_features_no_batch[num_special_tokens:, :]
            patch_features_only = patch_features_only.reshape(14, 14, -1)
            
            row_features.append(patch_features_only)
        
        all_features.append(torch.stack(row_features))
    
    # Stack all (14, 14, feature_dim) features to create (70, 70, feature_dim) tensor
    dense_features = torch.stack(all_features)
    
    return dense_features


def compute_dense_cosine_similarity(dense_features, dense_features_2, pixel_position, original_size):
    """
    Compute cosine similarity using dense sliding window features.
    dense_features: (70, 70, feature_dim) tensor
    pixel_position: (y, x) coordinates in original image
    original_size: (height, width) of original image
    """
    # Convert pixel position to dense feature coordinates
    h, w = original_size
    dense_h, dense_w = dense_features.shape[:2]
    
    # Map pixel position to dense feature grid
    dense_y = int((pixel_position[0] / h) * dense_h)
    dense_x = int((pixel_position[1] / w) * dense_w)
    
    # Ensure coordinates are within bounds
    dense_y = min(dense_y, dense_h - 1)
    dense_x = min(dense_x, dense_w - 1)
    
    # Get the feature vector at the specified position
    pixel_features = dense_features[dense_y, dense_x, :]  # (feature_dim,)
    
    # Normalize the pixel feature
    pixel_features_norm = F.normalize(pixel_features.unsqueeze(0), p=2, dim=1)  # (1, feature_dim)
    
    # Reshape dense_features_2 for normalization: (70, 70, feature_dim) -> (70*70, feature_dim)
    dense_features_2_flat = dense_features_2.reshape(-1, dense_features_2.shape[-1])  # (70*70, feature_dim)
    dense_features_2_norm = F.normalize(dense_features_2_flat, p=2, dim=1)  # (70*70, feature_dim)
    
    # Compute cosine similarity: (1, feature_dim) @ (feature_dim, 70*70) = (1, 70*70)
    similarity = torch.mm(pixel_features_norm, dense_features_2_norm.t()).squeeze(0)  # (70*70,)
    
    # Reshape similarity to match the dense feature grid
    similarity = similarity.reshape(dense_h, dense_w)
    
    # Interpolate similarity back to original image size
    similarity = similarity.unsqueeze(0).unsqueeze(0)  # (1, 1, 70, 70)
    similarity = F.interpolate(similarity, size=(h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)  # (h, w)
    
    return similarity


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
    patch_features = F.interpolate(patch_features, size=(h, w), mode='bilinear', align_corners=False).squeeze(0)  # (C, h, w)
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
    patch_features_2 = F.interpolate(patch_features_2, size=(h, w), mode='bilinear', align_corners=False).squeeze(0)  # (C, h, w)
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

# Extract dense features using sliding window approach
print("Extracting dense features from reference image...")
dense_features = extract_sliding_window_features(image, model, processor)
print(f"Dense features shape: {dense_features.shape}")

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

    # Extract dense features using sliding window approach
    print(f"Extracting dense features from {file_name}...")
    dense_features_2 = extract_sliding_window_features(image_2, model, processor)
    print(f"Dense features shape: {dense_features_2.shape}")

    vis_image_2 = np.array(image_2).astype(np.uint8)

    # find the most similar patch using dense features
    for point_id, point in enumerate(points):
        u = point[0]
        v = point[1]
        similarity = compute_dense_cosine_similarity(dense_features, dense_features_2, (v, u), (h, w))

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

