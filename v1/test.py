"""This module is used to test the Srnet model."""
from glob import glob
import os
import torch
import numpy as np
import imageio.v2 as io
from model.model import Srnet

TEST_BATCH_SIZE = 40

# Expand home directory
COVER_PATH = os.path.expanduser("~/data/GBRASNET/BOSSbase-1.01-div/cover/val")
STEGO_PATH = os.path.expanduser("~/data/GBRASNET/BOSSbase-1.01-div/stego/S-UNIWARD/0.4bpp/stego/val")
COVER_PATH = os.path.expanduser("/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOWS2/cover/test")
STEGO_PATH = os.path.expanduser("/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOWS2/stego/WOW/0.2bpp/test")
CHKPT = "./checkpoints/SRNet_model_weights.pt"

# Load image paths - try multiple extensions
cover_image_names = (glob(f"{COVER_PATH}/*.pgm") +
                     glob(f"{COVER_PATH}/*.png") +
                     glob(f"{COVER_PATH}/*.jpg"))

stego_image_names = (glob(f"{STEGO_PATH}/*.pgm") +
                     glob(f"{STEGO_PATH}/*.png") +
                     glob(f"{STEGO_PATH}/*.jpg"))

# Sort for consistency
cover_image_names = sorted(cover_image_names)
stego_image_names = sorted(stego_image_names)

print(f"Found {len(cover_image_names)} cover images")
print(f"Found {len(stego_image_names)} stego images")

if len(cover_image_names) == 0 or len(stego_image_names) == 0:
    print(f"\nERROR: No images found!")
    print(f"Cover path: {COVER_PATH}")
    print(f"Stego path: {STEGO_PATH}")
    print(f"Cover path exists: {os.path.exists(COVER_PATH)}")
    print(f"Stego path exists: {os.path.exists(STEGO_PATH)}")
    exit(1)

# Setup device and model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = Srnet().to(device)
ckpt = torch.load(CHKPT, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()  # Set to evaluation mode

# Prepare batch tensor
images = torch.empty((TEST_BATCH_SIZE, 1, 256, 256), dtype=torch.float)

test_accuracy = []

# Calculate number of batches
num_batches = min(len(cover_image_names), len(stego_image_names)) // (TEST_BATCH_SIZE // 2)
print(f"Processing {num_batches} batches...")

with torch.no_grad():  # Disable gradient computation for testing
    for batch_idx in range(0, len(cover_image_names), TEST_BATCH_SIZE // 2):
        # Get cover and stego batches
        cover_batch = cover_image_names[batch_idx : batch_idx + TEST_BATCH_SIZE // 2]
        stego_batch = stego_image_names[batch_idx : batch_idx + TEST_BATCH_SIZE // 2]

        # Skip if we don't have enough images for a full batch
        if len(cover_batch) < TEST_BATCH_SIZE // 2 or len(stego_batch) < TEST_BATCH_SIZE // 2:
            break

        # Interleave cover and stego images
        batch = []
        batch_labels = []

        for i in range(TEST_BATCH_SIZE // 2):
            batch.append(stego_batch[i])
            batch_labels.append(1)
            batch.append(cover_batch[i])
            batch_labels.append(0)

        # Load images into tensor
        for i in range(TEST_BATCH_SIZE):
            images[i, 0, :, :] = torch.tensor(io.imread(batch[i]), dtype=torch.float)

        image_tensor = images.to(device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

        # Get predictions
        outputs = model(image_tensor)
        prediction = outputs.data.max(1)[1]

        # Calculate accuracy
        accuracy = (prediction.eq(batch_labels.data).sum().float() * 100.0 / batch_labels.size(0))
        test_accuracy.append(accuracy.item())

        # Print progress
        if (len(test_accuracy)) % 10 == 0:
            print(f"  Processed {len(test_accuracy)} batches, current accuracy: {accuracy:.2f}%")

# Print final results
if len(test_accuracy) > 0:
    avg_accuracy = sum(test_accuracy) / len(test_accuracy)
    print(f"\n{'='*60}")
    print(f"Test Results")
    print(f"{'='*60}")
    print(f"Total batches processed: {len(test_accuracy)}")
    print(f"Average test accuracy: {avg_accuracy:.2f}%")
    print(f"{'='*60}")
else:
    print("\nERROR: No batches were processed. Check your image paths and batch size.")
