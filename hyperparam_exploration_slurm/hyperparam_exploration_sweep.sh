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
        job_script="${dir_name}/job_script.sh"

        cat > "$job_script" <<EOL
#!/bin/bash
#SBATCH -A bii_dsc_community
#SBATCH -p bii-gpu
#SBATCH --gres=gpu:v100:4
#SBATCH -N 4
#SBATCH --cpus-per-task=8
#SBATCH --mem=375G
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --job-name=edm2_pm${P_mean}_ps${P_std}
#SBATCH --output=${dir_name}/sigma03_job_output_${P_mean}_ps${P_std}.log
#SBATCH --error=${dir_name}/sigma03_job_error_${P_mean}_ps${P_std}.log


# Record the start time
start_time=\$(date +%s)

# Load the necessary modules
module load anaconda
conda activate edm2

# Calculate total_nimg
number_of_epochs=100
n_samples=127488  # Replace with your dataset size
batch_size=1280
total_nimg=\$((number_of_epochs * n_samples))

# Set master address and port for distributed training
MASTER_ADDR=\$(scontrol show hostname \$SLURM_NODELIST | head -n 1)
MASTER_PORT=29502  # Choose an available port

# Export environment variables
export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE=\$((SLURM_NNODES * 4))  # Nodes * GPUs per node

nodes=( \$(scontrol show hostnames \$SLURM_JOB_NODELIST) )
nodes_array=(\${nodes[@]})
head_node=\${nodes_array[0]}
head_node_ip=\$(srun --nodes=1 --ntasks=1 -w "\$head_node" hostname --ip-address)

# Set status, snapshot, and checkpoint intervals
status_nimg=\$((batch_size * 10))
snapshot_nimg=\$((n_samples * 10))
checkpoint_nimg=\$snapshot_nimg

echo "MASTER_ADDR=\$MASTER_ADDR"
echo "MASTER_PORT=\$MASTER_PORT"
echo "WORLD_SIZE=\$WORLD_SIZE"
echo "Rendezvous Endpoint: \${head_node_ip}:29502"

echo "Node IP: \${head_node_ip}"
export LOGLEVEL=INFO

for node in "\${nodes_array[@]}"; do
    echo "Testing connectivity to \$node"
    ping -c 4 "\$node"
done

# Run the training script using torchrun
srun torchrun \
    --nnodes \$SLURM_NNODES \
    --rdzv_id \$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=\${head_node_ip}:29502 \
    --nproc_per_node=4 \
    /scratch/tc2fh/edm2_diffusion/edm2_sigma03/train_edm2.py \
    --outdir="${dir_name}/training-runs" \
    --data=/scratch/tc2fh/Angiogenesis_Generative_data/edm2_singlevalue_labeled_singlechannel.zip \
    --preset=edm2-img64-s \
    --batch-gpu=4 \
    --batch=\$batch_size \
    --duration=\$total_nimg \
    --status=\$status_nimg \
    --snapshot=\$snapshot_nimg \
    --checkpoint=\$checkpoint_nimg \
    --cond=True \
    --P_mean="${P_mean}" \
    --P_std="${P_std}"
    
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

        dir_name="sigma03_sweep/pm${P_mean}_ps${P_std}"
        job_script="${dir_name}/job_script.sh"

        # Submit the job script
        sbatch "$job_script"

    done
done
