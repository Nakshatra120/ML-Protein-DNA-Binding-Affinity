#!/bin/bash
#SBATCH --job-name=naccess
#SBATCH --account=jyu20_lab
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:00:00

# Path to the NACCESS executable (solvent accessible surface area calculator)
naccess='/dfs6/pub/calmasri/naccess/Naccess/naccess'

# Loop through all PDB files matching the patterns:
#   - complex*.pdb : full protein–DNA complexes
#   - prot*.pdb    : protein-only structures
#   - dna*.pdb     : DNA-only structures
for pdb_file in complex*.pdb prot*.pdb dna*.pdb; 
do
    echo "Running naccess on $pdb_file"
    # Compute solvent-accessible surface areas (ASA).
    # Outputs:
    #   $pdb_file.rsa  → relative ASA
    #   $pdb_file.asa  → atomic ASA
    #   $pdb_file.log  → summary log
    $naccess $pdb_file
done

echo "All PDB files processed."
