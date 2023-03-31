#!/usr/bin/env bash
#SBATCH  --mail-type=ALL                 # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH  --output=%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --gres=gpu:titan_xp:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID
#
# binary to execute
set -o errexit

source /itet-stor/owendu/net_scratch/miniconda3/bin/activate env
srun python paraphrases150k/helper.py
echo finished at: `date`
exit 0;
