#!/bin/bash
#SBATCH --mail-type=END                 # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --job-name=sim                  # create a short name for your job
#SBATCH --nodes=1                       # node count
#SBATCH --gres=gpu:4   # titan_rtx & geforce_rtx_3090 & tesla_v100 & geforce_rtx_2080_ti & rtx_a6000
#SBATCH --cpus-per-task=3               # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G               # total memory per node (4 GB per cpu-core is default)

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID
#
# binary to execute
set -o errexit

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export CUDA_VISIBLE_DEVICES=0,1,2,3

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=4

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=4

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
source /itet-stor/owendu/net_scratch/miniconda3/bin/activate simcse
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path roberta-base \
    --train_file  data/wiki_vicuna.txt \
    --output_dir result/vicuna/roberta_15_nodp \
    --num_train_epochs 15 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --dropout_val 0.0 \
    --paraphrase_mode \
    --paraphrase_file /itet-stor/owendu/net_scratch/Paraphrase/FastChat/pps/pp.json \
    "$@"
echo finished at: `date`
exit 0;

