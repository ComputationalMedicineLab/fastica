#!/bin/bash
# Prior to running, make sure (by any convenient method) that XT.npy is
# situated in the correct directory. `XT` is expected to be a C-contiguous
# ndarray of shape [n_features, n_samples].
function run_model() {
    fastica whiten "$1/XT.npy" \
        --log-file=$4 \
        -X "$1/X1_$2.npy" \
        -K "$1/K_$2.npy" \
        -M "$1/X_mean.npy" \
        -c $3

    mkdir -p "$1/checkpoints_$2"

    fastica run "$1/X1_$2.npy" \
        --log-file=$4 \
        -W "$1/W_$2.npy" \
        --w-init "$1/W_init_$2.npy" \
        --max-iter=200 \
        --tol=1e-3 \
        --checkpoint-iter=1 \
        --checkpoint-dir="$1/checkpoints_$2" \
        --checkpoint-iter-format='03d' \
        --cwork=320

    fastica recover \
        --log-file=$4 \
        --X1-file="$1/X1_$2.npy" \
        --W-file="$1/W_$2.npy" \
        --K-file="$1/K_$2.npy" \
        --S-file="$1/S_$2.npy" \
        --A-file="$1/A_$2.npy" \
        --scale --alpha=2.0 --sign-flip \
        --factors-file="$1/factors_$2.npy"
}
run_model "./data/models/model_2k" "2k" 2000 "./logs/run_model_2k.log"
