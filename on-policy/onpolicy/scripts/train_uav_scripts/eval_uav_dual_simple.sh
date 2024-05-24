#!/bin/sh
# exp param
env="UAV"
scenario="N_v_Simple"
algo="rmappo" # "mappo" "ippo"
exp="check"

# uav group param
num_agents=3 # two pursuers vs 1 evader

# train param
num_env_steps=0
# episode_length=400

# echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

CUDA_VISIBLE_DEVICES=0 python3 ../eval/eval_uav_dual_simple.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --n_rollout_threads 32 --ppo_epoch 15 --num_mini_batch 2 \
--save_interval 200000 --log_interval 1 --n_eval_rollout_threads 32 --eval_episodes 100 --hidden_size 128 \
--user_name "leakycauldron" --wandb_name "first-trial" --use_wandb --use_eval \
--d_k 8 --n_head 4 --encoder_hidden_size 32 --attn_size 24 --pursuer_num 1 --evader_num 1 --recurrent_N 2 \
--file_path "/home/qiyuan/workspace/plot/20240201" --file_name "track_3D_2_optimal_" --layer_N 2 --encoder_layer_N 1  --model_dir "/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/scripts/train_uav_scripts/wandb/run-20240129_032808-34hrqhhk/files" \
# --use_calu_prob
