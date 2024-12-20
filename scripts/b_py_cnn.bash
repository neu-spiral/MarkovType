#!/bin/bash
for seed in "0"  "1" "2" "3" "4"
do
   for model_type in "cnn-1d" "cnn-2d"
   do
       work=scripts/
       cd $work
       sbatch execute_cnn.bash $seed $model_type
   done
done
