#!/bin/bash

#SBATCH --job-name=mmpbsa
#SBATCH --output=out.txt
#SBATCH --partition=standard
#SBATCH --account=jyu20_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=1G

# Initialize Conda for bash and activate the MMGBSA environment
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate mmgbsa_ml

# Run gmx_MMPBSA in parallel across 10 MPI ranks
mpirun -np 10 gmx_MMPBSA MPI \
    -O \                                     # Overwrite existing output files if present
    -i mmgbsa_strain.in \                    # Input control file
    -cs npt_nowat.tpr \                      # GROMACS portable run file (no-water)
    -cp topol_nowat.top \                    # Topology for no-water system
    -ci index_nowat.ndx \                    # Index file with groups for decomposition (1,4)
    -cg 1 4 \                                # Groups to compute binding between (e.g., protein=1, DNA=4)
    -ct npt_whole_nowat.xtc \                # Trajectory file (no-water)
    -o FINAL_RESULTS_strain.dat \            # Overall binding energy output (plain text)
    -eo FINAL_RESULTS_strain.csv \           # Overall binding energy output (CSV)
    -do FINAL_RESULTS_DECOMP_strain.dat \    # Decomposed per-residue energies (plain text)
    -deo FINAL_RESULTS_DECOMP_strain.csv \   # Decomposed per-residue energies (CSV)
    &> out.txt &                             # Redirect all output to log file and run in background

# Wait for the background MMGBSA process to complete
wait $!
