#!/bin/bash
#SBATCH -J TEST                      # job name
#SBATCH -o stdout.txt      # output and error file name (%j expands to jobID)
#SBATCH -p X1
#SBATCH -N 1               # total number of nodesmpi tasks requested
#SBATCH -n 8              # total number of mpi tasks requested


python ../../run.py \
       max_iters=30 \
