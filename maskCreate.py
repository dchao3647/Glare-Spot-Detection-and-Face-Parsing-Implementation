import torch
import os
from torch import nn
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Convenience expression for automatically determining the device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Load models
image_processor = AutoImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
model.to(device)

# Input an image and resize it to 360x480
image_path = "input1.jpg" 
image = Image.open(image_path)
#resized_image = image.resize((480, 360))

# Save the resized input image
#resized_image_path = "input3.jpg"
#resized_image.save(resized_image_path)
#print(f"Resized input image saved as: {resized_image_path}")

# Run inference on the resized image
inputs = image_processor(images=image, return_tensors="pt").to(device)
outputs = model(**inputs)
logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

# resize output to match input image dimensions
upsampled_logits = nn.functional.interpolate(logits,
                size=image.size[::-1], # H x W
                mode='bilinear',
                align_corners=False)

# Get label masks
labels = upsampled_logits.argmax(dim=1)[0]

# Move to CPU to visualize in matplotlib
labels_viz = labels.cpu().numpy()

# Define the label for the nose region
nose_label = 2

# Create a mask for the nose region
nose_mask = labels_viz == nose_label

# Convert the original resized image into a numpy array
image_np = np.array(image)

# Identify bright spots on the nose
brightness_threshold = 145.5
bright_spots_mask = np.all(image_np > brightness_threshold, axis=-1) & nose_mask

# Convert the mask to uint8
bright_spots_mask_uint8 = (bright_spots_mask * 255).astype(np.uint8)

# Save the final mask image
final_mask_path = "mask1.jpg"
cv2.imwrite(final_mask_path, bright_spots_mask_uint8)
print(f"Final mask image saved as: {final_mask_path}")

# Display the resized nose mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(nose_mask, cmap='gray')
plt.title("Nose Mask")

# Display the bright spots mask
plt.subplot(1, 2, 2)
plt.imshow(bright_spots_mask_uint8, cmap='gray')
plt.title("Bright Spots Mask")

plt.show()



