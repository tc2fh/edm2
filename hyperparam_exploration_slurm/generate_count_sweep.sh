#!/bin/bash

# Define the ranges for P_mean and P_std
P_mean_values=(-1.6 -0.5333 0.5333 1.6)   # np.linspace(-1.6, 1.6, 4)
P_std_values=(1 1.6667 2.3333 3)          # np.linspace(1, 3, 4)

# Base output directory
BASE_OUTPUT_DIR="/scratch/tc2fh/edm2_diffusion/sigma03_cond_pixelspace_1channel_sweep"

# Loop over all combinations of P_mean and P_std
for P_mean in "${P_mean_values[@]}"; do
    for P_std in "${P_std_values[@]}"; do

        # Format the directory name, now including BASE_OUTPUT_DIR
        dir_name="${BASE_OUTPUT_DIR}/sigma03_sweep/pm${P_mean}_ps${P_std}"

        # Create the directory if it doesn't exist
        mkdir -p "$dir_name"

        # Create the Slurm job script within the directory
        job_script="${dir_name}/generate_classify.sh"

        cat > "$job_script" <<EOL
#!/bin/bash
#SBATCH -A bii_dsc_community
#SBATCH -p bii-gpu
#SBATCH --gres=gpu:v100:4
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --job-name=gencount_edm2_pm${P_mean}_ps${P_std}
#SBATCH --output=${dir_name}/gencount_job_output_${P_mean}_ps${P_std}.log


# Record the start time
start_time=\$(date +%s)

# Load the necessary modules
module load anaconda
conda activate edm2

# Generate images with trained model

MODEL_PATH="${dir_name}/training-runs/network-snapshot-0012748-0.100.pkl"
OUTDIR="${dir_name}/gen_images"

mkdir -p "\$OUTDIR"

torchrun --standalone --nproc_per_node=4 /scratch/tc2fh/edm2_diffusion/edm2/generate_images.py --net="\$MODEL_PATH" --outdir="\$OUTDIR" --seeds=0-19999 --steps=32

conda deactivate

#classify images with trained classifier

export CLASSIFIER_CKPT_PATH=/scratch/tc2fh/diffusion_classifier/testing_output_logs/lr00001/classifier/version_0/checkpoints/model-epoch=393-val_loss=0.02.ckpt
export IMG_DIR="\$OUTDIR"

conda activate diffusers_cuda124

# Path to the count images scripts
EXTERNAL_DIR="/scratch/tc2fh/edm2_diffusion/generate_images_bii_gpu/count_generated_images"

# Add the external directory to PYTHONPATH
export PYTHONPATH="\$EXTERNAL_DIR:$\PYTHONPATH"

chmod +x "\$EXTERNAL_DIR/read_classify_images_edm2.py"
python -u "\$EXTERNAL_DIR/read_classify_images_edm2.py"

# Record the end time
end_time=\$(date +%s)

# Calculate the elapsed time
elapsed=\$(( end_time - start_time ))

# Format and print the elapsed time in hh:mm:ss
hours=\$(( elapsed / 3600 ))
minutes=\$(( (elapsed % 3600) / 60 ))
seconds=\$(( elapsed % 60 ))
printf "Total job time: %02d:%02d:%02d\\n" "\$hours" "\$minutes" "\$seconds"

EOL

        # Make the job script executable
        chmod +x "$job_script"

    done
done

# After creating all job scripts, loop through and submit them
for P_mean in "${P_mean_values[@]}"; do
    for P_std in "${P_std_values[@]}"; do

        dir_name="${BASE_OUTPUT_DIR}/sigma03_sweep/pm${P_mean}_ps${P_std}"
        job_script="${dir_name}/generate_classify.sh"

        # Submit the job script
        sbatch "$job_script"

    done
done