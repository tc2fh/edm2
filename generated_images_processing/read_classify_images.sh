#!/bin/bash

#SBATCH -A bii_dsc_community
#SBATCH -p bii-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00

# used to check the number of classes in a zarr dataset using the trained classifier

export CLASSIFIER_CKPT_PATH=/scratch/tc2fh/diffusion_classifier/testing_output_logs/lr00001/classifier/version_0/checkpoints/model-epoch=393-val_loss=0.02.ckpt
export IMG_DIR=/scratch/tc2fh/Angiogenesis_Generative_data/all_images.zarr.zip

module load anaconda
conda activate diffusers_cuda124
chmod +x read_classify_images_edm2.py
python -u read_classify_images_edm2.py