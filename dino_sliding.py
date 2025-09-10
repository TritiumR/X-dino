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


parser = argparse.ArgumentParser(description="Test DINOv3")
parser.add_argument("--file_path", type=str, help="Path to the image file")
parser.add_argument("--folder_path", type=str, help="Path to the image folder")
parser.add_argument("--crop_size", type=int, help="Crop size")
parser.add_argument("--stride", type=int, help="Stride")
args = parser.parse_args()

# Load DINOv3 model and processor directly
print("Loading DINOv3 model...")
# model_name = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
# model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
# feature_dim = 768
model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
feature_dim = 1024
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(
    model_name, 
)

# def slide_inference(
#     inputs: torch.Tensor,
#     segmentation_model: nn.Module,
#     decoder_head_type: str = "linear",
#     n_output_channels: int = 256,
#     crop_size: Tuple = (512, 512),
#     stride: Tuple = (341, 341),
#     num_max_forward: int = 1,
# ):
#     """Inference by sliding-window with overlap.
#     If h_crop > h_img or w_crop > w_img, the small patch will be used to
#     decode without padding.
#     Args:
#         inputs (tensor): the tensor should have a shape NxCxHxW,
#             which contains all images in the batch.
#         segmentation_model (nn.Module): model to use for evaluating on dense tasks.
#         n_output_channels (int): number of output channels
#         crop_size (tuple): (h_crop, w_crop)
#         stride (tuple): (h_stride, w_stride)
#     Returns:
#         Tensor: The output results from model of each input image.
#     """
#     h_stride, w_stride = stride
#     h_crop, w_crop = crop_size
#     batch_size, C, h_img, w_img = inputs.shape
#     if h_crop > h_img and w_crop > w_img:  # Meaning we are doing < 1.0 TTA
#         h_crop, w_crop = min(h_img, w_img), min(h_img, w_img)
#     assert batch_size == 1  # As of now, the code assumes that a single image is passed at a time at inference time
#     h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
#     w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
#     preds = inputs.new_zeros((1, n_output_channels, h_img, w_img)).cpu()
#     count_mat = inputs.new_zeros((1, 1, h_img, w_img)).to(torch.int8).cpu()
#     for h_idx in range(h_grids):
#         for w_idx in range(w_grids):
#             y1 = h_idx * h_stride
#             x1 = w_idx * w_stride
#             y2 = min(y1 + h_crop, h_img)
#             x2 = min(x1 + w_crop, w_img)
#             y1 = max(y2 - h_crop, 0)
#             x1 = max(x2 - w_crop, 0)
#             crop_img = inputs[:, :, y1:y2, x1:x2]
#             crop_pred = segmentation_model.predict(crop_img, rescale_to=crop_img.shape[2:])
#             preds += F.pad(crop_pred, (int(x1), int(preds.shape[-1] - x2), int(y1), int(preds.shape[-2] - y2))).cpu()
#             count_mat[:, :, y1:y2, x1:x2] += 1
#             del crop_img, crop_pred
#     # Optional buffer to ensure each gpu does the same number of operations for sharded models
#     for _ in range(h_grids * w_grids, num_max_forward):
#         dummy_input = inputs.new_zeros((1, C, h_crop, w_crop))
#         _ = segmentation_model.predict(dummy_input, rescale_to=dummy_input.shape[2:])
#     assert (count_mat == 0).sum() == 0
#     return preds / count_mat


def extract_sliding_window_features(image, model, processor, crop_size=args.crop_size, stride=args.stride):
    """
    Extract features using sliding window approach.
    Takes 1/5 of the image dimensions and creates overlapping patches.
    Returns dense features of shape (70, 70, feature_dim).
    """
    NUM_SPECIAL_TOKENS = 5
    # Get image dimensions
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Calculate number of patches
    num_patches_h = h // stride + 1
    num_patches_w = w // stride + 1
    
    # Initialize feature tensor
    all_features = torch.zeros((h, w, feature_dim))
    feature_count = torch.zeros((h, w, 1))
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Calculate crop coordinates
            h1 = i * stride
            w1 = j * stride

            h2 = min(h1 + crop_size, h)
            w2 = min(w1 + crop_size, w)

            h1 = max(h2 - crop_size, 0)
            w1 = max(w2 - crop_size, 0)
            
            # Extract crop
            crop = image.crop((w1, h1, w2, h2))
            
            # Process patch
            inputs = processor(images=crop, return_tensors="pt").to(model.device)
            
            with torch.inference_mode():
                outputs = model(**inputs)

            crop_features = outputs.last_hidden_state

            # Extract patch features (exclude special tokens)
            crop_features_no_batch = crop_features.squeeze(0)  # Remove batch dimension
            crop_features_only = crop_features_no_batch[NUM_SPECIAL_TOKENS:, :]
            crop_features_only = crop_features_only.reshape(14, 14, -1)

            # interpolate to crop size
            crop_features_only = crop_features_only.permute(2, 0, 1).unsqueeze(0)  # (1, C, 14, 14)
            crop_features_only = F.interpolate(crop_features_only, size=(crop_size, crop_size), mode='bilinear', align_corners=False).squeeze(0)  # (C, crop_size, crop_size)
            crop_features_only = crop_features_only.permute(1, 2, 0)  # (crop_size, crop_size, C)

            # print(crop_features_only.shape)

            all_features[h1:h2, w1:w2, :] += crop_features_only
            feature_count[h1:h2, w1:w2] += 1
    
    assert (feature_count == 0).sum() == 0

    # print(all_features.shape)
    # print(feature_count.shape)
    all_features = all_features / feature_count
    
    return all_features


def compute_dense_cosine_similarity(dense_features, dense_features_2, pixel_position):
    """
    Compute cosine similarity using dense sliding window features.
    dense_features: (h, w, feature_dim) tensor
    pixel_position: (y, x) coordinates in original image
    original_size: (height, width) of original image
    """
    # Convert pixel position to dense feature coordinates
    h, w = dense_features.shape[:2]
    
    # Get the feature vector at the specified position
    pixel_features = dense_features[pixel_position[0], pixel_position[1], :]  # (feature_dim,)
    
    # Normalize the pixel feature
    pixel_features_norm = F.normalize(pixel_features.unsqueeze(0), p=2, dim=1)  # (1, feature_dim)
    
    # Reshape dense_features_2 for normalization
    dense_features_2_flat = dense_features_2.reshape(-1, dense_features_2.shape[-1])
    dense_features_2_norm = F.normalize(dense_features_2_flat, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.mm(pixel_features_norm, dense_features_2_norm.t()).squeeze(0)
    
    # Reshape similarity to match the dense feature grid
    similarity = similarity.reshape(h, w)
    
    return similarity


print("Processing frames and extracting features...")

image = load_image(args.file_path)

# Extract dense features using sliding window approach
print("Extracting dense features from reference image...")
dense_features = extract_sliding_window_features(image, model, processor)
print(f"Dense features shape: {dense_features.shape}")

vis_image = np.array(image).astype(np.uint8)

h, w = vis_image.shape[:2]

point_file = args.file_path.replace(".png", ".pkl")

points = np.array(pickle.load(open(point_file, "rb"))).astype(np.int32)
# points = np.array([[0, 658, 471], [0, 1018, 328], [0, 1075, 480]])

noise_to_points = np.random.random(size=(len(points), 2)) * 0
points[:, 1:] = points[:, 1:] + noise_to_points

points = points[:, 1:]

# print(points)

output_folder = os.path.join("output", args.file_path.split("/")[-1].split(".")[0], f"sliding_{args.crop_size}_{args.stride}_{noise_to_points.sum()}")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
for point_id, point in enumerate(points):
    u = point[0]
    v = point[1]
    vis_image = cv2.circle(vis_image, (u, v), 5, (0, 255 - point_id * 255 // len(points), point_id * 255 // len(points)), -1)

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
        similarity = compute_dense_cosine_similarity(dense_features, dense_features_2, (v, u))

        # print(similarity.shape)

        most_similar_patch = np.argmax(similarity)

        # print(most_similar_patch)

        # Convert flattened index back to 2D coordinates
        u = int(most_similar_patch % w)  # x-coordinate (column)
        v = int(most_similar_patch // w)  # y-coordinate (row)

        # print(u, v)

        vis_image_2 = cv2.circle(vis_image_2, (u, v), 5, (0, 255 - point_id * 255 // len(points), point_id * 255 // len(points)), -1)

    vis_image_2 = cv2.cvtColor(vis_image_2, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(output_folder, file_name.replace(".png", f"_corr.png")), vis_image_2)

