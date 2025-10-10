#!/bin/bash
set -x
export PROJECT_NAME='PRIS'

export EXPERIMENT_NAME=grpo_distill_r1_1.5_16k-n8-$(date +%Y-%m-%d_%H-%M-%S)

export TRAIN_DATASET='your_train_dataset_path'
export TEST_DATASET='your_test_dataset_path'

export ACTOR_MODEL_PATH='your_model_path'
export PROJECT_PATH='your_project_path'
export CKPT_PATH=${PROJECT_PATH}/checkpoints



rollout_mode="async"
rollout_name="vllm" # sglang or vllm
return_raw_chat="True"

if [ $RANK -eq 0 ]; then
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.return_raw_chat=$return_raw_chat \
    ++data.gen_batch_size=384 \
    data.train_batch_size=256 \
    data.val_batch_size=4096 \
    data.max_prompt_length=1024 \
    data.max_response_length=31744 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
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
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    trainer.logger=['console','tensorboard','swanlab'] \
    trainer.balance_batch=True \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=20 \
    trainer.default_local_dir="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME"
    "$@" 
fi