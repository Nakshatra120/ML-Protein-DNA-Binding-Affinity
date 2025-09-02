#!/bin/bash
#
#SBATCH --job-name=npt
#SBATCH --output=out_npt_%j.txt 
#SBATCH --account=jyu20_lab_gpu
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1

# Set environment variable for GROMACS forcefield location
export GMXLIB=/dfs6/pub/calmasri/ForceFields

# Load the GROMACS 2022.1 module with CUDA support
module load gromacs/2022.1/gcc.8.4.0-cuda.11.7.1

# Print the SLURM_JOBID of the previous job (if any)
echo "Previous slurm jobID: ${SLURM_JOBID_PREV}"

# Define filenames based on the previous job ID
prev_out_file="out_npt_${SLURM_JOBID_PREV}.txt"
resubmit_script="submit_npt.sh"

# Logic for automatic resubmission on preemption or time limit
if [ -z $SLURM_JOBID_PREV ]; then
    # First invocation: no previous job ID present
    echo "First iteration, submit dependency"
    sbatch --dependency=afternotok:${SLURM_JOBID} \
           --export=SLURM_JOBID_PREV=${SLURM_JOBID} \
           ${resubmit_script}
else
    # From second invocation onward:
    if [ -f "${prev_out_file}" ] && (grep -q "PREEMPTION" "${prev_out_file}" \
       || grep -q "TIME LIMIT" "${prev_out_file}"); then
        # If preemption or time limit exceeded detected, resubmit
        echo "Preemption or Time limit exceeded, resubmitting"
        sbatch --dependency=afternotok:${SLURM_JOBID} \
               --export=SLURM_JOBID_PREV=${SLURM_JOBID} \
               ${resubmit_script}
    else
        # If failure was not due to preemption or time limit, abort
        echo "Error is not due to preemption or time limit, exit"
        exit 1
    fi
fi

# Execute the production NPT run: restart from checkpoint if available
if [ -a npt.cpt ]; then
    # Restart from an existing checkpoint
    gmx mdrun -ntmpi 1 -ntomp 10 -deffnm npt \
              -cpi npt.cpt \
              -nb gpu -pme gpu -bonded cpu
else
    # Prepare a new NPT run (generate .tpr) using NPT0 outputs
    gmx grompp -f npt.mdp -c npt0.gro -r npt0.gro \
               -t npt0.cpt -p topol.top -o npt.tpr
    # Execute the new NPT run
    gmx mdrun -ntmpi 1 -ntomp 10 -deffnm npt \
              -nb gpu -pme gpu -bonded cpu
fi
