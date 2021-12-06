#!/bin/sh
#SBATCH --verbose
#SBATCH -p aquila
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=12GB
#SBATCH --mail-type=END # select which email types will be sent
#SBATCH --mail-user=yh2689@nyu.edu # put your email here if you want emails        

#SBATCH --output=log/runh_%A.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID make sure you create the log folder
#SBATCH --error=log/runh_%A.err

sleep $(( (RANDOM%10) + 1 ))


module purge                      
module load anaconda3 glew/1.13 glfw/3.3 glog/0.3.3 mesa/19.0.5 llvm/7.0.1 gcc/7.3 
module load anaconda3 cuda/11.1.1 

nvidia-smi
nvcc --version
cd /gpfsnyu/scratch/yh2689/REDQ-student/plot_utils

source deactivate
source /gpfsnyu/packages/anaconda3/5.2.0/bin/activate keith

echo ${SLURM_ARRAY_TASK_ID}
python plot_1-1-3.py
echo "DONE"