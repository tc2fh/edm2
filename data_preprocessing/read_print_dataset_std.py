#calculates the mean and std of the dataset

import zipfile
import numpy as np
from PIL import Image
from tqdm import tqdm

# Path to the zip file containing images
zip_filename = '/scratch/tc2fh/Angiogenesis_Generative_data/edm2_singlevalue_labeled_singlechannel.zip'

# Variables to accumulate statistics
n_pixels = 0
mean = 0
m2 = 0  # Sum of squares of differences from the mean (for variance calculation)

# Open the zip file and process each image incrementally
with zipfile.ZipFile(zip_filename, 'r') as zipf:
    for img_filename in tqdm(zipf.namelist()):
        if img_filename.endswith('.png'):  # Only process PNG images
            with zipf.open(img_filename) as img_file:
                img = Image.open(img_file).convert('L')
                img_array = np.array(img, dtype=np.float32).flatten() / 255.0  # Normalize to [0, 1]
                
                # Update incremental mean and variance calculation
                for pixel in img_array:
                    n_pixels += 1
                    delta = pixel - mean
                    mean += delta / n_pixels
                    delta2 = pixel - mean
                    m2 += delta * delta2

# Final standard deviation calculation
variance = m2 / n_pixels
std_dev = np.sqrt(variance)

print(f"Mean: {mean}, Standard Deviation: {std_dev}")
