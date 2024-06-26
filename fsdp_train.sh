#!/bin/bash
#SBATCH --job-name=test-fsdp-new
#SBATCH --output=log/fsdp-%j.txt
#SBATCH --error=log/fsdp-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
# activate the environment
source /gpfs/u/home/LMCG/LMCGnngn/scratch/miniconda3x86/etc/profile.d/conda.sh
conda activate mllm


echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"

RANDOM=$$
DIV=1000
OFFSET=24000
MASTER_PORT=$(($RANDOM%$DIV+$OFFSET))
export TORCHELASTIC_LOG_LEVEL=DEBUG

#export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
#echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR

# set environment variable 
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
NODE_RANK=${SLURM_PROCID}

# if ']' is the last character of the node list
SLURM=${SLURM_NODELIST:0:3}
if [ $SLURM == "dcs" ]; then
    # change to your local director
    export PYTHONUSERBASE="/gpfs/u/home/LMCG/LMCGnngn/scratch/.local"
fi

# print the ip address, node list and node rank
echo $SLURM_NODELIST
echo $NODE_RANK
echo "NODE NUMBER:"$SLURM_NNODES

# run the training script
NUM_GPUS_PER_NODE=$(echo "$SLURM_JOB_GPUS" | tr ',' '\n' | wc -l)
echo $NUM_GPUS_PER_NODE

if [ $SLURM_NNODES -gt 1 ]; then
    CMD="torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$NUM_GPUS_PER_NODE --node_rank=$NODE_RANK
    --rdzv-id=$RANDOM --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT"
else
    CMD="torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --master_port=$MASTER_PORT"
fi

echo $CMD

srun $CMD \
fsdp_train.py \
--folder retrieval_attention_6_all \
--lr=1e-6 \
--num_epochs=115 \
--batch_size=1 \
--save_interval=1
