#!/bin/bash
#set a job name  
#SBATCH --job-name=bci_cnn
#################  
#a file for job output, you can check job progress
#SBATCH --output=bci_cnn.out
#################
# a file for errors from the job
#SBATCH --error=bci_cnn.err
#################
#time you think you need; default is one day
#in minutes in this case, hh:mm:ss
#SBATCH --time=23:59:59
#################
#number of tasks you are requesting
#SBATCH -N 1
#SBATCH --exclusive
#################
#partition to use
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=150Gb
#################
#number of nodes to distribute n tasks across
#################

python train.py --seeds $1 --model_type $2
#python evaluate.py --model_type $1 --without-threshold
#python parse_results.py --model_type $1
