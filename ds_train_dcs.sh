#!/bin/bash
#SBATCH --job-name=test-ds
#SBATCH --output=log/dcs_scannet-%j.txt
#SBATCH --error=log/dcs_scannet-%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:6
#SBATCH --nodes=24
# activate the environment
# source /gpfs/u/home/LMCG/LMCGnngn/scratch/miniconda3x86/etc/profile.d/conda.sh
#source ~/.bashrc_dcs
#conda activate /gpfs/u/home/LMCG/LMCGnngn/scratch/miniconda3/envs/jc-eqa
#conda activate jc-eqa
#conda activate eqa


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

# TODO: set up deepspeed args
# use train_micro_batch_size_per_gpu to set up dataloader
# zero3 offload has unsolved problems
config_json="./ds_cfg/zero2.json"
ZERO_STAGE=0
DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    "
    #--zero-stage ${ZERO_STAGE} \
    #--deepspeed-activation-checkpointing \
    #"
export CUDA_LAUNCH_BLOCKING=1


if [ $SLURM_NNODES -gt 1 ]; then
    CMD="torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$NUM_GPUS_PER_NODE --node_rank=$NODE_RANK
    --rdzv_id=$RANDOM --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT"
else
    CMD="torchrun --nproc_per_node=$NUM_GPUS_PER_NODE --master_port=$MASTER_PORT"
fi

echo $CMD

# always set every choice to true to achieve peak GPU memory
srun $CMD \
deepspeed_train.py \
--folder ckpts/scannet_finetuned \
--random_permute \
--lr=4e-6 \
--num_epochs=6 \
--batch_size=1 \
--patch_size=1 \
--visual_feature_size=3 \
--max_length=4508 \
$DEEPSPEED_ARGS \
--egocentric_views \
--lora_enable \
--augment_question \
--num_egocentric_views=5 \
--prefiltering \
--filter_coeff=0.4 \
--top_k_categories=10 \


# automically evaluate after training current model

#train_job_id=$SLURM_JOB_ID

# evaluate first half ckpts
# eval_job_id1=$(sbatch --dependency=afterok:$train_job_id eval_sh/ds_eval.sh)
# echo "Submitted evaluation job with ID $eval_job_id1"

# eval_job_id2=$(sbatch --dependency=afterok:$train_job_id eval_sh/ds_eval_5.sh)
# echo "Submitted evaluation job with ID $eval_job_id2"