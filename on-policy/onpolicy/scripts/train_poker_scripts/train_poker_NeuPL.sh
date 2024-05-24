#!/bin/sh
# exp param
env="LEDUC_POKER"
scenario="POKER"
algo="mappo" # "mappo" "ippo"
exp="neupl_test"

# uav group param
num_agents=3 # two pursuers vs 1 evader

# train param
num_env_steps=800
# episode_length=400

# echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

CUDA_VISIBLE_DEVICES=0 python3 ../train/train_poker_NeuPL.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --n_rollout_threads 200 --ppo_epoch 15 --num_mini_batch 2 \
--save_interval 10 --log_interval 1 --n_eval_rollout_threads 2 --eval_episodes 100 --use_proper_time_limits \ #--use_render \
--user_name "leakycauldron" --wandb_name "first-trial" --gamma 1.0 --entropy_coef 0.05 --sigma_layer_N 2 --layer_N 2 --encoder_layer_N 1 \
--hidden_size 128 --encoder_hidden_size 32 --attn_size 24 --recurrent_N 2 --sigma_encoder_layer_N 1 \
--critic_lr 0.00001 --lr 0.00001 --total_round 100 --use_mix_policy --population_size 33 --eval_episode_num 30 --channel_interval 1 --use_fast_update --update_T 1000 \
--role_number 0 --policy_backup_dir "/home/qiyuan/workspace/policy_backup/neupl_poker_20240416" #--role_number 0 #--use_warmup False
