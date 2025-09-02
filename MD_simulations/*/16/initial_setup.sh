#!/bin/bash
#
#SBATCH --job-name=equil0
#SBATCH --output=equil0.txt 
#SBATCH --account=JYU20_LAB
#
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5000
#SBATCH --cpus-per-task=10
#SBATCH --time=3-00:00:00


# Load the GROMACS 2022.1 module (compiled with GCC 8.4.0 and CUDA 11.7.1)
# Provides the gmx command and associated libraries for MD simulations
module load gromacs/2022.1/gcc.8.4.0-cuda.11.7.1

# Point GROMACS to your locally installed force‐field parameter library
export GMXLIB=/dfs6/pub/calmasri/ForceFields

# Submit the equilibration job after this job completes successfully
sbatch --dependency=afterok:${SLURM_JOBID} submit_equil.sh

# Find the input PDB file matching MycMax*.pdb in the current directory tree
pdb_file=$(find . -type f -name "MycMax*.pdb")
# Define the name for the processed PDB output
output_pdb="prot_processed.pdb"

# Use awk to remove the first three phosphate atoms (P, O1P, O2P) from chains A and B
awk '
    BEGIN { a_count = 0; b_count = 0; }
    {
        # For chain A: skip first 3 occurrences of P, O1P, or O2P
        if ($5 == "A" && a_count < 3 && ($3 == "P" || $3 == "O1P" || $3 == "O2P")) {
            a_count++;
            next;
        }
        # For chain B: skip first 3 occurrences of P, O1P, or O2P
        if ($5 == "B" && b_count < 3 && ($3 == "P" || $3 == "O1P" || $3 == "O2P")) {
            b_count++;
            next;
        }
        # Print all other lines unchanged
        print;
    }
' "$pdb_file" > "$output_pdb"

# Generate topology and coordinate files, merging all chains into one protein group, adding hydrogens with TIP3P water model
# '-renum yes' ensures atom renumbering; '-ignh' ignores existing hydrogens in the input
echo 1 | gmx pdb2gmx \
    -f $output_pdb \
    -o protein_processed.gro \
    -ss \
    -merge all \
    -water tip3p \
    -renum yes \
    -ignh &
wait $!

# Define a new simulation box for the protein, centering it and ensuring at least 1.2 nm distance to the box edge
echo DNA | gmx editconf \
    -f protein_processed.gro \
    -o protein_newbox.gro \
    -princ yes \
    -c \
    -d 1.2
wait $!

# Solvate the protein in the new box, updating the topology file
gmx solvate \
    -cp protein_newbox.gro \
    -o protein_solv.gro \
    -p topol.top
wait $!

# Prepare for ion addition: generate a tpr file for the ion placement step, allowing up to 1 warning
gmx grompp \
    -f ions.mdp \
    -c protein_solv.gro \
    -p topol.top \
    -o ions.tpr \
    -maxwarn 1

# Add ions to neutralize the system and reach 0.15 M NaCl concentration
echo SOL | gmx genion \
    -s ions.tpr \
    -o protein_solv_ions.gro \
    -p topol.top \
    -pname NA \
    -nname CL \
    -neutral \
    -conc 0.15 &
wait $!
