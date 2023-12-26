#!/usr/bin/env bash

#SBATCH --job-name=comp
#SBATCH --partition=instruction

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --output=comp.out
#SBATCH --error=comp.err
#SBATCH --time=0-00:00:10

cd $SLURM_SUBMIT_DIR

nvcc master.cu bfs.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o cudapath
