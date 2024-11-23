'''
reads all png images in a directory, classifies it, and counts the number of images in each class

before classifying, it converts rgb to gray and thresholds by the mean

for use with images generated from the edm2 fork, latent diffusion models where training data is converted to rgb images
'''

#%% imports
import os
import torch
import numpy as np
from train_classifier import ClassificationEfficientNet
from typing import List, Optional, Tuple, Union
import PIL
import logging
import zarr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

#%%  definitions

classifier_checkpoint_path = os.getenv('CLASSIFIER_CKPT_PATH', r"C:\Users\TC\Documents\LearningPython\Angiogenesis_ML_testing\runs\diffusion_classifier\500_epoch_train\lr00001\classifier\version_0\checkpoints\model-epoch=393-val_loss=0.02.ckpt") #0.956 test accuracy
img_dir = os.getenv('IMG_DIR')

device_str = 'cuda' # 'cuda' or 'cpu'

def load_classifier_model(checkpoint_path, device='cuda'):
    '''load trained classifier model and move to device'''
    Classifier_Model = ClassificationEfficientNet()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    Classifier_Model.load_state_dict(checkpoint['state_dict'])  # Load the Classifier_Model's state_dict from the checkpoint
    Classifier_Model.to(device)
    Classifier_Model.eval()
    return Classifier_Model

def pass_through_classifier(image, model, device='cuda'):
    '''pass image through classifier model'''
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device) #unsqueeze twice to give batch and channel dimensions from 256x256 image
    output = model(image)
    predicted_label = output.argmax(dim=1).item()  # Get the index of the argmax and convert to a Python scalar
    return predicted_label

def main():
    logging.info("Starting the classification")

    # Load the trained classifier model
    classifier = load_classifier_model(classifier_checkpoint_path, device=device_str)

    with torch.no_grad():

        predicted_labels = []
        for file in tqdm(os.listdir(img_dir)):
            if file.endswith(".png"):
                image_path = os.path.join(img_dir, file)
                with PIL.Image.open(image_path) as image:
                    # Convert the new image to a numpy array and then to grayscale
                    new_image_array = np.array(image)
                    gray_new_image = np.mean(new_image_array, axis=2).astype(np.uint8)

                    # Threshold the grayscale new image by its mean value
                    new_gray_mean = gray_new_image.mean()
                    thresholded_new_gray_image = gray_new_image > new_gray_mean

                predicted_label = pass_through_classifier(thresholded_new_gray_image.astype(np.float32), classifier, device=device_str)
                predicted_labels.append(predicted_label)

        # for each unique label, count the number of images
        logging.info(f'number of predicted labels: {len(predicted_labels)}')
        unique_labels, counts = np.unique(predicted_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            logging.info(f"Label {label} has {count} images")




if __name__ == "__main__":
    main()