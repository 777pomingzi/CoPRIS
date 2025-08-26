#!/bin/bash
#SBATCH --job-name=hbx_block                    # 作业名称
#SBATCH --output=output_%j.log      # 标准输出和错误日志文件名 (%j 表示作业ID)
#SBATCH --error=error_%j.log        # 错误日志文件名
#SBATCH --account=test1267
#SBATCH --partition=TEST1_XCJ                  # 分区名称      
#SBATCH --nodelist=g[28]      
#SBATCH --gres=gpu:8                      # 每个节点请求 8 块 GPU
#SBATCH --ntasks=1                        # 总任务数
#SBATCH --cpus-per-task=64                # 每个任务分配的CPU核心数
#SBATCH --mem=1000G                       # 分配的内存大小
#SBATCH --nodes=1                         # 使用 4 个节点
#SBATCH --ntasks-per-node=1               # 每个节点启动 1 个任务


cat <<EOF > run_training.sh
#!/bin/bash
set -x
export VLLM_USE_V1=1
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1
unset ROCR_VISIBLE_DEVICES

GPUS_PER_NODE=8

export PROJECT_NAME='async-partial'
export DSR_DATA_DIR=/home/test1267/test-6/hbx/rllm/data
export OR1_DATA_DIR=/home/test1267/test-6/hbx/datasets/rl_data/or1_data
export BASE_DATA_DIR=/home/test1267/test-6/hbx/datasets/rl_data

# TODO:
export EXPERIMENT_NAME=grpo_dapo_distill_r1_1p5_16k-n8-\$(date +%Y-%m-%d_%H-%M-%S)
SWANLAB_API_KEY=HtpjItuIsLT7SGwM4bQmB
SWANLAB_LOG_DIR=/home/test1267/test-6/qzk/verl-main-08-18/swanlab/\${EXPERIMENT_NAME}
SWANLAB_MODE="cloud"

swanlab login --relogin HtpjItuIsLT7SGwM4bQmB
# TODO:
# export TRAIN_DATASET=\$BASE_DATA_DIR/dapo-math-17k.parquet
export TRAIN_DATASET=/home/test1267/test-6/qzk/Datasets/DAPO/DAPO.parquet
export TEST_MATH=\$DSR_DATA_DIR/deepscaler_math.parquet
# export TEST_AIME24=\$OR1_DATA_DIR/aime24.parquet
# export TEST_AIME25=\$OR1_DATA_DIR/aime25.parquet
export TEST_AIME=\$DSR_DATA_DIR/deepscaler_aime.parquet
# export TEST_AIME=\$BASE_DATA_DIR/aime-2024.parquet
export TEST_DATASET="['\$TEST_AIME','\$TEST_MATH']"

# TODO:
export ACTOR_MODEL_PATH=/home/test1267/test-6/qzk/PLM/DeepSeek-R1-Distill-Qwen-1.5B
# export ACTOR_MODEL_PATH=/home/test1267/test-6/qzk/PLM/OpenMath-Nemotron-1.5B
# export ACTOR_MODEL_PATH=/home/test1267/test-6/qzk/PLM/qwen3-1.7b
export PROJECT_PATH=/home/test1267/test-6/qzk/verl-main-08-18
export PARALLEL_SIZE=1
export CKPT_PATH=\${PROJECT_PATH}/checkpoints
export NCCL_DEBUG=WARN
export WANDB_API_KEY='7e69f789501e2f5153bf315454c1f1a414b06c55'
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=offline
export WANDB_DIR=\${PROJECT_PATH}/wandb/
export TENSORBOARD_DIR=\${PROJECT_PATH}/tensorboard/\$PROJECT_NAME/\$EXPERIMENT_NAME
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

RANK=\${SLURM_PROCID}

if [ \$RANK -eq 0 ]; then
    rm -f /home/test1267/test-6/hbx/ray_head_ip.txt
else
    sleep 60
fi

ray stop
sleep 30

HEAD_NODE=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)

if [ \$RANK -eq 0 ]; then
    ray start --head --port=6379 --num-gpus=8 --include-dashboard=false > /home/test1267/test-6/hbx/ray_head_ip.txt 2>&1 &
    echo "Ray Head \$(hostname)"
fi

sleep 30

while [ ! -f /home/test1267/test-6/hbx/ray_head_ip.txt ]; do
    sleep 2
done

MASTER_IP=\$(grep 'Local node IP' /home/test1267/test-6/hbx/ray_head_ip.txt | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+')
export RAY_ADDRESS="\${MASTER_IP}:6379"
echo "Ray Head IP: \$RAY_ADDRESS"

sleep 60

if [ \$RANK != 0 ]; then
    ray start --address="\$RAY_ADDRESS" --num-gpus=\${GPUS_PER_NODE} --block
    echo "Worker \$(hostname) joined Ray Head"
fi

sleep 60

ray status

WORLD_SIZE=\${SLURM_NTASKS}

rollout_mode="async"
rollout_name="vllm" # sglang or vllm
return_raw_chat="True"

if [ \$RANK -eq 0 ]; then
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="\$TRAIN_DATASET" \
    data.val_files="\$TEST_DATASET" \
    data.return_raw_chat=\$return_raw_chat \
    ++data.gen_batch_size=64 \
    data.train_batch_size=64 \
    data.val_batch_size=4096 \
    data.max_prompt_length=1024 \
    data.max_response_length=15360 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=\$ACTOR_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    ++actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    ++actor_rollout_ref.rollout.filter_groups=False \
    ++actor_rollout_ref.rollout.partial_rollout_pool_size=512 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=\$rollout_name \
    actor_rollout_ref.rollout.mode=\$rollout_mode \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.max_new_tokens=31744 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    trainer.logger=['console','tensorboard','swanlab'] \
    trainer.balance_batch=True \
    trainer.project_name=\$PROJECT_NAME \
    trainer.experiment_name=\$EXPERIMENT_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=\${WORLD_SIZE} \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=20 \
    trainer.default_local_dir="\$CKPT_PATH/\$PROJECT_NAME/\$EXPERIMENT_NAME"
    "\$@" 
fi
EOF


chmod +x run_training.sh

srun --mpi=pmi2 --gres=gpu:8 --ntasks=${SLURM_NTASKS} --ntasks-per-node=1 ./run_training.sh