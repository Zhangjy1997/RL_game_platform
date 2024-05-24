#!/bin/sh
# exp param
env="UAV"
scenario="N_v_interaction"
algo="rmappo" # "mappo" "ippo"
exp="check_ag3"

# uav group param
num_agents=3 # two pursuers vs 1 evader

# train param
num_env_steps=24000000
# episode_length=400

# echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

CUDA_VISIBLE_DEVICES=0 python3 ../train/train_uav_interaction.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --n_rollout_threads 32 --ppo_epoch 15 --num_mini_batch 2 \
--save_interval 10 --log_interval 1 --n_eval_rollout_threads 2 --eval_episodes 100 --use_proper_time_limits \ #--use_render \
--user_name "leakycauldron" --wandb_name "first-trial" --d_k 8 --n_head 4 --layer_N 2 --encoder_layer_N 1 --hidden_size 128 --encoder_hidden_size 32 --attn_size 24 \
--critic_lr 0.000025 --lr 0.000025 --total_round 2 --use_mixer --model_dir "/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/scripts/train_uav_scripts/wandb/run-20231230_051738-2t6wy4tk/files"

