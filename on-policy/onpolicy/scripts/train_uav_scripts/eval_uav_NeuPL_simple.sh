#!/bin/sh
# exp param
env="UAV"
scenario="N_v_Simple"
algo="mappo" # "mappo" "ippo"
exp="check"

# uav group param
num_agents=3 # two pursuers vs 1 evader

# train param
num_env_steps=0
# episode_length=400

# echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

CUDA_VISIBLE_DEVICES=0 python3 ../eval/eval_uav_simple_PL.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --n_rollout_threads 1 --ppo_epoch 15 --num_mini_batch 2 \
--save_interval 10 --log_interval 1 --n_eval_rollout_threads 2 --eval_episodes 100 --use_proper_time_limits \ #--use_render \
--user_name "leakycauldron" --wandb_name "first-trial" --d_k 8 --n_head 4 --sigma_layer_N 2 --layer_N 2 --encoder_layer_N 1 --hidden_size 128 --encoder_hidden_size 32 --attn_size 24 --recurrent_N 2 \
--critic_lr 0.00001 --lr 0.00001 --total_round 30 --use_mix_policy --population_size 5 --eval_episode_num 20 --channel_interval 1 --use_wandb \
--file_path "/home/qiyuan/workspace/plot/20240425"  --model_dir "/home/qiyuan/workspace/policy_backup/PSRO_uav_NE_20240417/round44" --track_n 25 \
--use_track_recorder
