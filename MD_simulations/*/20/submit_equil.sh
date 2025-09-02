#!/bin/bash
#
#SBATCH --job-name=equil
#SBATCH --output=out_equil.txt 
#SBATCH --account=jyu20_lab_gpu
#
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=5000
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00

# Load the GROMACS 2022.1 module (compiled with GCC 8.4.0 and CUDA 11.7.1)
# Provides the gmx command and associated libraries for MD simulations
module load gromacs/2022.1/gcc.8.4.0-cuda.11.7.1

# Point GROMACS to your locally installed force‐field parameter library
export GMXLIB=/dfs6/pub/calmasri/ForceFields

# Submit the NVT equilibration step as a dependent SLURM job,
# to run only after the current job (e.g. minimization) completes successfully.
sbatch --dependency=afterok:${SLURM_JOBID} submit_nvt.sh


# PREPROCESSING: Generate the binary input file (.tpr) for energy minimization
gmx grompp -f minim.mdp -c protein_solv_ions.gro -p topol.top -o em.tpr -r protein_solv_ions.gro

# RUNNING MINIMIZATION: Perform steepest-descent energy minimization
gmx mdrun -ntmpi 1 -ntomp 10 -v -deffnm em