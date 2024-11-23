#!/bin/bash
#SBATCH -A bii_dsc_community
#SBATCH -p standard                   # Partition (queue) name
#SBATCH -N 1                             # Number of nodes
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --mem=64G                       # Total memory
#SBATCH --ntasks-per-node=1               # Number of tasks per node
#SBATCH --time=24:00:00                   # Time limit hrs:min:sec
#SBATCH --job-name=edm2_generate          # Job name

module load anaconda
conda activate edm2

chmod +x prep_labeled_dataset_singlevalue_classlabel.py
python -u prep_labeled_dataset_singlevalue_classlabel.py