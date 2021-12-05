#!/bin/sh
#SBATCH --verbose
#SBATCH -p aquila
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=12GB
#SBATCH --mail-type=END # select which email types will be sent
#SBATCH --mail-user=yh2689@nyu.edu # put your email here if you want emails        

#SBATCH --array=0-11 # here the number depends on number of jobs in the array, 0-4 means 0, 1, 2, 3, 4, a total of 5 jobs
#SBATCH --output=log/1-2-3/runh_%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID make sure you create the log folder
#SBATCH --error=log/1-2-3/runh_%A_%a.err

sleep $(( (RANDOM%10) + 1 ))

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module purge                      
module load anaconda3 glew/1.13 glfw/3.3 glog/0.3.3 mesa/19.0.5 llvm/7.0.1 gcc/7.3 
module load anaconda3 cuda/11.1.1 

nvidia-smi
nvcc --version
cd /gpfsnyu/scratch/yh2689/REDQ-student

source deactivate
source /gpfsnyu/packages/anaconda3/5.2.0/bin/activate keith

echo ${SLURM_ARRAY_TASK_ID}
python train_redq_sac_exp1-2-3.py --setting ${SLURM_ARRAY_TASK_ID}
echo "DONE"