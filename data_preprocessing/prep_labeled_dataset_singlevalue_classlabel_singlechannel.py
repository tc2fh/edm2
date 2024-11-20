# %% load all data and save to an uncompressed zip file with a class label
'''
load all data from the .zarr.zip files and save to an uncompressed zip file with a class label
class labels are single integer values that correspond to the classes of the trained classifier

note: this keeps images as single channel images from the original dataset
'''

import os
import io
import zarr
import numpy as np
import zipfile
import json
import random
from PIL import Image
from tqdm import tqdm

def load_images(zipstore_path):
    """
    Load all images from a specific zarr zipstore.
    """
    with zarr.open(store=zarr.storage.ZipStore(zipstore_path, mode="r")) as root:
        images = root["fgbg"][:]
    return images

def extract_params_from_foldername(filename):
    """
    Extract parameters from the filename.

    filename convention is f'contact_{contact_param}_decay_{decay_param}_simulation_{simulation_replicate}.zarr.zip'

    returns tuple of (contact, decay)
    """
    contact = int(float(filename.split("_")[1]))
    decay = float(filename.split("_")[3])
    return (contact, decay)

# Path to .zarr.zip file
data_path = '/scratch/tc2fh/Angiogenesis_Generative_data/' # path to Angiogenesis_Generative_data, where each folder contains simulation replicates of a specific parameter set

# Create a new zip file to store individual images
zip_filename = 'edm2_singlevalue_labeled_singlechannel.zip'
output_zip_dir = '/scratch/tc2fh/Angiogenesis_Generative_data/'
output_zip_path = os.path.join(output_zip_dir, zip_filename)

# Label mapping
decay_values = [0.05, 0.1875, 0.325, 0.4625, 0.6]
contact_values = [0, 5, 10, 15, 20]
label_mapping = {(contact, decay): idx for idx, (contact, decay) in enumerate([(c, d) for c in contact_values for d in decay_values])}

# List to store image filenames and labels
images_and_labels = []

# Collect all images and labels
for folder in tqdm(os.listdir(data_path)):
    folder_path = os.path.join(data_path, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".zarr.zip"):
                file_path = os.path.join(folder_path, file)
                images = load_images(file_path)
                
                # Extract parameters from folder name
                contact, decay = extract_params_from_foldername(file)
                label = label_mapping[(contact, decay)]
                
                # Convert images to uint8 format (0 and 255) for saving as PNG
                images_uint8 = (images * 255).astype(np.uint8)
                
                for idx, img in enumerate(images_uint8):
                    # Convert the numpy array to a PIL Image
                    pil_img = Image.fromarray(img)
                    
                    # Save image to a bytes buffer
                    img_buffer = io.BytesIO()
                    pil_img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    # Create a filename (e.g., image_00000000.png)
                    img_filename = f'image_{len(images_and_labels):08d}.png'
                    
                    # Add the image filename and label to the list
                    images_and_labels.append((img_filename, label, img_buffer))


# Shuffle the dataset to prevent bias during training
random.shuffle(images_and_labels)

# Write shuffled images and labels to the zip file
with zipfile.ZipFile(output_zip_path, 'w') as zipf:
    for img_filename, label, img_buffer in images_and_labels:
        # Write the image to the zip file
        zipf.writestr(img_filename, img_buffer.read())

# Create a list of lists for shuffled labels
shuffled_labels_list = [[img_filename, label] for img_filename, label, _ in images_and_labels]

# Save the shuffled labels list as a JSON file in the zip folder
labels_json_path = os.path.join(output_zip_dir, 'dataset.json')
with open(labels_json_path, 'w') as json_file:
    json.dump({'labels': shuffled_labels_list}, json_file)


# Add the JSON file to the zip file
with zipfile.ZipFile(output_zip_path, 'a') as zipf:
    zipf.write(labels_json_path, 'dataset.json')

# Remove the JSON file from the directory after adding it to the zip
os.remove(labels_json_path)

print(f"Images and labels have been saved to {output_zip_path}")