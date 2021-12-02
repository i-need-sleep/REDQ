#!/bin/sh
#SBATCH --verbose
#SBATCH -p aquila,parallel
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=12GB
#SBATCH --mail-type=ALL # select which email types will be sent
#SBATCH --mail-user=netid@nyu.edu # put your email here if you want emails

#SBATCH --array=0-11 # here the number depends on number of jobs in the array, 0-4 means 0, 1, 2, 3, 4, a total of 5 jobs
#SBATCH --output=log/runh_%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID make sure you create the log folder
#SBATCH --error=log/runh_%A_%a.err

# #####################################################
# #SBATCH --gres=gpu:1 # uncomment this line to request for a gpu if your program uses gpu
#SBATCH --constraint=cpu # use this if you want to only use cpu

sleep $(( (RANDOM%10) + 1 ))

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load anaconda3
source deactivate
source deactivate
source activate rl

echo ${SLURM_ARRAY_TASK_ID}
python train_redq_sac_exp1-1-1.py --setting ${SLURM_ARRAY_TASK_ID}
