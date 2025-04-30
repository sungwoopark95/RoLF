#!/bin/bash

seeds=$(python3 get_primes.py --nums 100 200)
K_values=(20 30 40)
k=35
d=17  # Define the d_values array with 1, (k/2), and (k-1)
feat_bound=1
sigmas=(0.04 0.07 0.1)
param_bound=1
T=1200
delta=0.0001
trials=5
p=0.6
cases=(1 2 3)
explores=("double" "triple" "quad")
today=$(date +"%Y-%m-%d")

# Iterate over each value in seeds, p_values, d_values, and delta_values and run the command
IFS=','
for seed in $seeds;
do
    for sigma in "${sigmas[@]}"
    do
        for explore in "${explores[@]}"
        do
            for case in "${cases[@]}"
            do
                for K in "${K_values[@]}"
                do
                    python main.py --trials $trials --horizon $T --arms $K \
                    --latent_dim $k --dim $d --seed "$seed" --feat_dist gaussian \
                    --feat_feature_bound 1 --feat_bound_method clipping \
                    --feat_bound_type lsup --feat_disjoint --reward_std $sigma \
                    --param_dist uniform --param_uniform_rng -0.5 0.5 \
                    --param_bound 1 --param_disjoint --param_bound_type l1 \
                    --p $p --delta $delta --explore --init_explore $explore \
                    --case $case --date $today
                done
            done
        done
    done
done