#!/usr/bin/env bash
#SBATCH --array=3-9
#SBATCH  --mail-type=END                 # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH  --output=%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=15
#SBATCH --mem=30G
#SBATCH --nodelist=arton[05]

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID
#
# binary to execute
set -o errexit

source /itet-stor/owendu/net_scratch/miniconda3/bin/activate env
srun python paraphrase150k.py
echo finished at: `date`
exit 0;
