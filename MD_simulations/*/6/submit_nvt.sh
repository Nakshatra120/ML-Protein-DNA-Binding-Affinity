#!/bin/bash
#
#SBATCH --job-name=nvt
#SBATCH --output=out_nvt_%j.txt  # Using %j to include job ID in output filename
#SBATCH --account=jyu20_lab_gpu
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

# Set environment variable for GROMACS forcefield location
export GMXLIB=/dfs6/pub/calmasri/ForceFields

# Load the GROMACS 2022.1 module with CUDA support
module load gromacs/2022.1/gcc.8.4.0-cuda.11.7.1

# Print the SLURM_JOBID of the previous job (if any)
echo "Previous slurm jobID: ${SLURM_JOBID_PREV}"

# Define filenames based on the previous job ID
prev_out_file="out_nvt_${SLURM_JOBID_PREV}.txt"
resubmit_script="submit_nvt.sh"
next_script="submit_npt0.sh"

# Logic for automatic resubmission on preemption:
if [ -z $SLURM_JOBID_PREV ]; then
    # First invocation: no previous job ID present
    echo "First iteration, submit dependency"
    # Submit the same NVT script again if this job fails (not OK),
    # passing current job ID for the next iteration’s check.
    sbatch --dependency=afternotok:${SLURM_JOBID} \
           --export=SLURM_JOBID_PREV=${SLURM_JOBID} \
           ${resubmit_script}
else
    # From second iteration onward:
    if [ -f "${prev_out_file}" ] && grep -q "PREEMPTION" "${prev_out_file}"; then
        # If the previous SLURM output file exists and contains "PREEMPTION",
        # assume the job was preempted and needs resubmission.
        echo "Preemption detected, resubmitting"
        sbatch --dependency=afternotok:${SLURM_JOBID} \
               --export=SLURM_JOBID_PREV=${SLURM_JOBID} \
               ${resubmit_script}
    else
        # If failure was not due to preemption, abort further resubmissions.
        echo "Error is not due to preemption, exit"
        exit 1
    fi
fi

# Regardless of preemption logic, submit the next step (NPT0) upon successful completion
sbatch --dependency=afterok:${SLURM_JOBID} ${next_script}

# Begin the NVT run: continue from checkpoint if available
if [ -a nvt.cpt ]; then
    # Restart a previous NVT run from checkpoint
    gmx mdrun -ntmpi 1 -ntomp 10 -deffnm nvt \
              -cpi nvt.cpt \
              -nb gpu -pme gpu -bonded cpu
else
    # Prepare a new NVT run (generate .tpr)
    gmx grompp -f nvt.mdp -c em.gro -r em.gro \
               -p topol.top -o nvt.tpr
    # Execute the new NVT run
    gmx mdrun -ntmpi 1 -ntomp 10 -deffnm nvt \
              -nb gpu -pme gpu -bonded cpu
fi
