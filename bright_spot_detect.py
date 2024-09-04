import torch
import os
from torch import nn
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from imutils import contours
from skimage import measure
import imutils

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Load models
image_processor = AutoImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
model.to(device)

# Input image
image_path = "input5.jpg"
image = Image.open(image_path)
image_np = np.array(image)

# Run inference on image
inputs = image_processor(images=image, return_tensors="pt").to(device)
outputs = model(**inputs)
logits = outputs.logits

# Resize output to match input image dimensions
upsampled_logits = nn.functional.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False)

# Get label masks
labels = upsampled_logits.argmax(dim=1)[0]
labels_viz = labels.cpu().numpy()

# Create a mask for the nose region
nose_label = 2
nose_mask = labels_viz == nose_label

# Convert image to grayscale and blur it
gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (21, 21), 0)

# Apply the nose mask to the grayscale image
masked_gray = cv2.bitwise_and(blurred, blurred, mask=nose_mask.astype(np.uint8))

# Threshold the image to reveal light regions in the blurred image
thresh = cv2.threshold(masked_gray, 180, 255, cv2.THRESH_BINARY)[1]

# Perform erosions and dilations to remove small blobs of noise
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)

# Perform connected component analysis
labels = measure.label(thresh, connectivity=2, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

# Filter components based on size
for label in np.unique(labels):
    if label == 0:
        continue
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    if numPixels > 300:
        mask = cv2.add(mask, labelMask)

# Find contours in the mask
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]

# Create a mask for bright spots
bright_spots_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

# Draw bright spots on the mask
for c in cnts:
    ((cX, cY), radius) = cv2.minEnclosingCircle(c)
    #cv2.circle(bright_spots_mask, (int(cX), int(cY)), int(radius), 255, -1)
    cv2.drawContours(bright_spots_mask, [c], 0, 255, -1)

# Inpaint the bright spots using OpenCV's INPAINT_TELEA
inpainted_image_np = cv2.inpaint(image_np, bright_spots_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Convert the result back to a PIL image
inpainted_image = Image.fromarray(inpainted_image_np)

# Save the image
final_image_path = "output_5.jpg"
inpainted_image.save(final_image_path)

# Display the result
plt.imshow(inpainted_image)
plt.axis('off')
plt.show()