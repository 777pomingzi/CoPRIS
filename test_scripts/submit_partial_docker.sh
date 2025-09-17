#!/bin/bash
#SBATCH --job-name=partial-rollout
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --account=test
#SBATCH --partition=TEST1
#SBATCH --nodelist=g[81,82]
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000G

set -euo pipefail

IMAGE="verlai/verl:latest"
SHM_SIZE="700g"

HOST_CODE="/home/test/test06/qzk/verl-partial-agent-loop"
HOST_DATA="/home/test1267/test-6/qzk/Datasets"
HOST_MODEL="/home/test1267/test-6/qzk/PLM/DeepSeek-R1-Distill-Qwen-1.5B"

CONT_CODE="/workspace/verl-partial-agent-loop"
CONT_DATA="/workspace/Datasets"
CONT_MODEL="/workspace/DeepSeek-R1-Distill-Qwen-1.5B"

HEAD_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')
echo "[INFO] Head node: $HEAD_NODE, IP: $HEAD_IP"

# 导出变量，传递给 bash -s
export HEAD_NODE HEAD_IP SHM_SIZE IMAGE HOST_CODE HOST_DATA HOST_MODEL CONT_CODE CONT_DATA CONT_MODEL

# 用 bash -s 传递脚本内容，不依赖 /tmp
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 bash -s << 'EOF'
set -euxo pipefail

if [ "$SLURMD_NODENAME" == "$HEAD_NODE" ]; then
  echo "[INFO] On HEAD node: $SLURMD_NODENAME"

  docker run --rm -i --gpus all --network host --shm-size "$SHM_SIZE" \
    -e CONT_CODE="$CONT_CODE" \
    -e CONT_DATA="$CONT_DATA" \
    -e CONT_MODEL="$CONT_MODEL" \
    -e SLURM_NNODES \
    -e HEAD_IP \
    -v "$HOST_CODE":"$CONT_CODE" \
    -v "$HOST_DATA":"$CONT_DATA" \
    -v "$HOST_MODEL":"$CONT_MODEL" \
    "$IMAGE" \
    bash -lc '
      set -euxo pipefail
      GPUS_PER_NODE=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 8)
      ray stop || true
      sleep 2
      ray start --head --port=6379 --dashboard-host=0.0.0.0
      sleep 5
      ray status

      export GPUS_PER_NODE=$GPUS_PER_NODE
      export WORLD_SIZE=${SLURM_NNODES:-1}
      export NCCL_DEBUG=WARN
      export TOKENIZERS_PARALLELISM=true
      export HYDRA_FULL_ERROR=1
      export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

      export CONT_CODE="${CONT_CODE}"
      export CONT_DATA="${CONT_DATA}"
      export CONT_MODEL="${CONT_MODEL}"

      export PROJECT_NAME="async-partial"
      export PROJECT_PATH="${CONT_CODE}"
      export CKPT_PATH="${PROJECT_PATH}/checkpoints"
      export WANDB_MODE=offline
      export WANDB_DIR="${PROJECT_PATH}/wandb/"
      export TENSORBOARD_DIR="${PROJECT_PATH}/tensorboard/${PROJECT_NAME}"

      export DATA_DIR="${CONT_DATA}"
      export ACTOR_MODEL_PATH="${CONT_MODEL}"
      export TRAIN_DATASET="${DATA_DIR}/deepscaler/deepscaler.parquet"
      export TEST_DATASET="[\"${DATA_DIR}/AIME24/test.parquet\",\"${DATA_DIR}/AIME25/test.parquet\"]"

      export EXPERIMENT_NAME="grpo_dapo_distill_r1_1p5_16k-n8-$(date +%Y-%m-%d_%H-%M-%S)"

      cd "${PROJECT_PATH}"

      python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        data.train_files="${TRAIN_DATASET}" \
        data.val_files="${TEST_DATASET}" \
        data.return_raw_chat=True \
        ++data.gen_batch_size=128 \
        data.train_batch_size=64 \
        data.val_batch_size=4096 \
        data.max_prompt_length=1024 \
        data.max_response_length=15360 \
        data.filter_overlong_prompts=True \
        data.truncation="error" \
        actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
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
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        ++actor_rollout_ref.rollout.filter_groups=False \
        ++actor_rollout_ref.rollout.partial_rollout_pool_size=1024 \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
        actor_rollout_ref.rollout.max_model_len=32768 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.multi_turn.format=hermes \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=32 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        trainer.critic_warmup=0 \
        algorithm.kl_ctrl.kl_coef=0.000 \
        trainer.logger=["console","tensorboard"] \
        trainer.balance_batch=True \
        trainer.project_name="${PROJECT_NAME}" \
        trainer.experiment_name="${EXPERIMENT_NAME}" \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=${GPUS_PER_NODE} \
        trainer.nnodes=${WORLD_SIZE} \
        trainer.save_freq=20 \
        trainer.test_freq=20 \
        trainer.default_hdfs_dir=null \
        trainer.total_epochs=20 \
        trainer.default_local_dir="${CKPT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

      ray stop || true
    '
else
  echo "[INFO] On WORKER node: $SLURMD_NODENAME"

  docker run --rm -i --gpus all --network host --shm-size "$SHM_SIZE" \
    -v "$HOST_CODE":"$CONT_CODE" \
    -v "$HOST_DATA":"$CONT_DATA" \
    -v "$HOST_MODEL":"$CONT_MODEL" \
    -e SLURM_NNODES \
    -e HEAD_IP \
    "$IMAGE" \
    bash -lc '
      set -euxo pipefail
      GPUS_PER_NODE=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 8)
      ray stop || true
      sleep 2
      ray start --address="'"$HEAD_IP"'":6379

      while ray status --address="'"$HEAD_IP"'":6379 >/dev/null 2>&1; do
        sleep 30
      done

      ray stop || true
    '
fi
EOF

echo "[INFO] Submitted. Check logs: output_${SLURM_JOB_ID}.log / error_${SLURM_JOB_ID}.log"