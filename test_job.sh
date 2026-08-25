#!/bin/bash
#SBATCH --job-name=test_account
#SBATCH --output=slurm_test_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=prod

echo "================================="
echo "Test Job Started"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "================================="
echo "If you can read this in the output file, your account is working perfectly!"
sleep 10
echo "Test Job Completed"
