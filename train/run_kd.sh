#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3,4,5,6

# "reverse" "forward" "tlsd" "cakld" "jsd" 
# Array of KD loss types
kd_loss_types=(
    "alpha_beta" "pearson_chi_square"
    "average_kl" "jensen_shannon" "cosine" "euclidean" "wasserstein" "focal"
    "contrastive" "soft_target_ce" "mutual_information" "pearson_correlation"
    "mmd" "emd" "hellinger" "total_variation" "renyi" "kl" "bhattacharyya"
    "angular" "mse" "huber" "taylor_softmax" "label_smoothing" "symmetric_kl"
    "jeffreys" "squared_hellinger" "triangular" "harmonic_mean" "jensen_difference"
    "k_divergence" "dice" "topsoe" "alpha" "beta" "gamma" "tweedie" "itakura_saito"
    "cauchy_schwarz" "energy" "gen_jensen_shannon" "bregman" "f_divergence"
    "chi_square" "log_euclidean" "jeffrey" "jensen_renyi" "tsallis" "sharma_mittal"
    "kernel_target_alignment" "max_correlation" "centered_kernel_alignment" "hsic"
    "fisher_rao" "geometric_mean" "quadratic_chi" "neyman_chi_square" "kullback"
    "resistor_average" "order_alpha" "ab_divergence"
)

# Loop through each KD loss type
for loss_type in "${kd_loss_types[@]}"
do
    echo "Running training with KD loss type: $loss_type"   
    
    bash train.sh ../data/generation/datasets/hf-llama-2-7b/mix_wiki_alpaca_8000.json \
    ./ckpts/hf-llama-2-7b/int2-g128/ ./logs/hf-llama-2-7b/int2-g128/ 4 "$loss_type" > ./logs/hf-llama-2-7b/int2-g128/kd_$loss_type.log 2>&1
    # Optional: add a delay between runs
    sleep 5
done

echo "All training runs completed."
