#!/bin/bash
for seed in "0" "1" "2" "3" "4"
do
   for lambda in "0.01" "0.05" "0" "1e-01"  "0.5"  "1"
   do
       for model_size in "small" 
       do
           for epoch in "200"
           do
               for reward in "InverseCube" "InverseSquare"  "Rational" "Linear" 
               do
                   work=scripts/
                   cd $work
                   sbatch execute.bash $seed $lambda $model_size $epoch $reward
               done
           done
       done
   done
done
