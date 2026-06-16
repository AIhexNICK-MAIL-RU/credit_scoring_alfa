#!/bin/zsh
cd /Users/a1/Desktop/5_перенос/magistr/2_new_mag/3_cred_scores
run() { echo "===== START FOLD $1 seed $2 $(date) ====="; python3 -u train_nn_sub.py $1 $2 750000; echo "===== END FOLD $1 $(date) ====="; }
run 2 42
run 3 99
run 4 7
echo "ALL FOLDS DONE"
