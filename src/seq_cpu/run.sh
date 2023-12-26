#!/usr/bin/env bash

#SBATCH --job-name=run
#SBATCH --partition=instruction

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --time=0-00:00:30

#cd $SLURM_SUBMIT_DIR

./pathfinding
