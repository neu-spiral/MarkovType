#!/bin/bash
#set a job name  
#SBATCH --job-name=bci_rnn
#################  
#a file for job output, you can check job progress
#SBATCH --output=bci_rnn.%j.out
#################
# a file for errors from the job
#SBATCH --error=bci_rnn.%j.err
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
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#################
#number of nodes to distribute n tasks across
#################

#python train_rnn.py --seeds $1 --lambda_loss $2 --model_size $3 --epochs $4 --reward $5
#python evaluate.py --rnn --lambda_loss $1 --model_size $2 --epochs $3 --reward $4 --without-threshold --validation
#python evaluate.py --rnn --lambda_loss $1 --model_size $2 --epochs $3 --reward $4 --without-threshold 
#python evaluate.py --rnn --lambda_loss $1 --model_size $2 --epochs $3 --reward $4
#python parse_results.py --rnn --lambda_loss $1 --model_size $2
#python analyze_results.py --rnn --lambda_loss $1 --model_size $2
