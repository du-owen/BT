#!/usr/bin/env bash
#SBATCH  --mail-type=END                 # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH  --output=%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID
#
# binary to execute
set -o errexit

source /itet-stor/owendu/net_scratch/miniconda3/bin/activate vicuna
python ../fastchat/eval/get_model_answer.py --model-id 1 \
    --model-path ../vicuna \
    --question-file /itet-stor/owendu/net_scratch/Paraphrase/FastChat/paraphrase/t.jsonl \
    --answer-file /itet-stor/owendu/net_scratch/Paraphrase/FastChat/paraphrase/answer.jsonl \
    --num-gpus 1
echo finished at: `date`
exit 0;



