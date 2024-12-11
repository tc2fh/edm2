#!/bin/bash
#SBATCH -A bii_dsc_community
#SBATCH -p bii-gpu                   # Partition (queue) name
#SBATCH --gres=gpu:v100:4
#SBATCH -N 1                             # Number of nodes
#SBATCH --cpus-per-task=8                 # Number of CPU cores per task
#SBATCH --mem=64G                       # Total memory
#SBATCH --ntasks-per-node=1               # Number of tasks per node
#SBATCH --time=24:00:00                   # Time limit hrs:min:sec
#SBATCH --job-name=edm2_generate_surrogate          # Job name
#sbatch --output=apply_surrogate.out

module load anaconda
conda activate edm2

MODEL_PATH="/scratch/tc2fh/edm2_diffusion/sigma03_cond_pixelspace_1channel_sweep/sigma03_sweep/pm0.5333_ps2.3333/training-runs/network-snapshot-0012748-0.100.pkl" #select the model to use
OUTDIR="/scratch/tc2fh/edm2_diffusion/sigma03_cond_pixelspace_1channel_sweep/apply_surrogate"

#create list of integers 0-24
for CLASSLABEL in {0..24}
do
    echo "Processing class $CLASSLABEL"
    IMGOUTDIR="$OUTDIR/class_$CLASSLABEL"
    mkdir -p "$IMGOUTDIR"

    # Record the start time
    start_time=$(date +%s)

    torchrun --standalone --nproc_per_node=4 /scratch/tc2fh/edm2_diffusion/edm2_sigma03/generate_images.py --net=$MODEL_PATH --class=$CLASSLABEL --outdir=$IMGOUTDIR --seeds=0 --steps=32

    # Record the end time
    end_time=$(date +%s)

    # Calculate the elapsed time
    elapsed=$(( end_time - start_time ))

    echo "Elapsed time: $elapsed seconds"
done