#!/bin/sh
# exp param
env="KUHN_POKER"
scenario="POKER"
algo="mappo" # "mappo" "ippo"
exp="psro_test"

# uav group param
num_agents=3 # two pursuers vs 1 evader

# train param
num_env_steps=150000
# episode_length=400

# echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

CUDA_VISIBLE_DEVICES=0 python3 ../train/train_poker_PSRO.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --n_rollout_threads 200 --ppo_epoch 15 --num_mini_batch 2 \
--save_interval 10 --log_interval 1 --n_eval_rollout_threads 2 --eval_episodes 100 --use_proper_time_limits \ #--use_render \
--user_name "leakycauldron" --wandb_name "first-trial" --gamma 1.0 --entropy_coef 0.05 --layer_N 3 --encoder_layer_N 1 --hidden_size 128 --recurrent_N 1 \
--critic_lr 0.00001 --lr 0.00001 --total_round 100 --use_calc_exploit --use_empty_policy --use_mix_policy --population_size 33 --eval_episode_num 20 \
--sub_round 2 --until_flat --policy_backup_dir "/home/qiyuan/workspace/policy_backup/PSRO_kuhn_poker_20240429" #--role_number 0 #--use_warmup False

