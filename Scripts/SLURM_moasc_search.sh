#!/bin/bash -l

moasc_script="./SLURM_moascgridscript.sh"

model_names=('MOASC4')
datasets=('MOA-MVSA-multiple')

# Slurm settings
partition="tier3"
time="04:00:00"
memory=92

# Range of seeds
start_seed=1
end_seed=25

echo "Submitting jobs for seeds $start_seed..$end_seed"
echo

first_run="true"

for model in "${model_names[@]}"; do
  for dset in "${datasets[@]}"; do
    for seed_val in $(seq "$start_seed" "$end_seed"); do

      # Export all variables
      export MODEL_NAME="$model"
      export dataset="$dset"
      export seed="$seed_val"
      export partition="$partition"
      export time="$time"
      export memory="$memory"
      
      export increment="$first_run"

      # Run the moasc script
      bash "$moasc_script"

      # For subsequent seeds subfolders won't increment
      first_run="false"
    done
  done
done

echo
echo "Submitted jobs for seeds in the range $start_seed..$end_seed!"