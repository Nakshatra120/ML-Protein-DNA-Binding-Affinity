#!/bin/bash
#
#SBATCH --job-name=npt0
#SBATCH --output=out_npt0_%j.txt 
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
prev_out_file="out_npt0_${SLURM_JOBID_PREV}.txt"
resubmit_script="submit_npt0.sh"
next_script="submit_npt.sh"

# Automatic resubmission logic on preemption:
if [ -z $SLURM_JOBID_PREV ]; then
    # First invocation: no previous job ID present
    echo "First iteration, submit dependency"
    # Submit this script again if it fails (not OK),
    # passing current job ID for the next iteration’s check.
    sbatch --dependency=afternotok:${SLURM_JOBID} \
           --export=SLURM_JOBID_PREV=${SLURM_JOBID} \
           ${resubmit_script}
else
    # From second iteration onward:
    if [ -f "${prev_out_file}" ] && grep -q "PREEMPTION" "${prev_out_file}"; then
        # If preemption is detected in previous output, resubmit
        echo "Preemption detected, resubmitting"
        sbatch --dependency=afternotok:${SLURM_JOBID} \
               --export=SLURM_JOBID_PREV=${SLURM_JOBID} \
               ${resubmit_script}
    else
        # If failure was not due to preemption, abort further resubmissions
        echo "Error is not due to preemption, exit"
        exit 1
    fi
fi

# Regardless of preemption, submit the next step (NPT) upon success
sbatch --dependency=afterok:${SLURM_JOBID} ${next_script}

# Begin the NPT0 run: continue from checkpoint if available
if [ -a npt0.cpt ]; then
    # Restart from an existing NPT0 checkpoint
    gmx mdrun -ntmpi 1 -ntomp 10 -deffnm npt0 \
              -cpi npt0.cpt \
              -nb gpu -pme gpu -bonded cpu
else
    # Prepare a new NPT0 run (generate .tpr) from NVT output
    gmx grompp -f npt0.mdp -c nvt.gro -r nvt.gro \
               -t nvt.cpt -p topol.top -o npt0.tpr
    # Execute the new NPT0 run
    gmx mdrun -ntmpi 1 -ntomp 10 -deffnm npt0 \
              -nb gpu -pme gpu -bonded cpu
fi
