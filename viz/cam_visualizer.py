import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

# Define synced image directory
SYNC_IMG_DIR = "data/july8/cam1_cal_data/"

# Get sorted list of PNG images that match the pattern
img_files = sorted(glob.glob(os.path.join(SYNC_IMG_DIR, "*.png")))

# Safety check: ensure at least one image is found
if not img_files:
    raise FileNotFoundError(f"No images found in {SYNC_IMG_DIR} with pattern 'image_*.png'")

# Load the first image
img = Image.open(img_files[0])

# Display the image
plt.imshow(img)
plt.axis('off')  # Optional: hide axis ticks
plt.title(os.path.basename(img_files[0]))  # Optional: show filename
plt.show()
