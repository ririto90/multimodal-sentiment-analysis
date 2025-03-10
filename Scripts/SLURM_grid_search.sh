#!/bin/bash -l

# Name of the sbatch script we want to submit many times
jobfile="./SLURM_gridscript.sh"

# Values for hyperparameters
model_names=('SIMPLE-F5')
fusion_models=('multiattfusion') # 'mfcchfusion2' 'mfcchfusion3' 'mfcchfusion4'
datasets=('MOA-MVSA-single') # 'mvsa-mts' 'mvsa-mts-balanced' 'mvsa-mts-oversampled' 'mvsa-mts-undersampled'
lr_vals=(0.001 0.0001 0.0005 0.00002 0.00005)
dr_vals=(0.5)
hidden_dims=(768)


echo
echo "Preparing to submit many jobs..."
echo

first_run="true"

# Create a grid search across all hyperparameter values
for dr in "${dr_vals[@]}"; do
    for lr in "${lr_vals[@]}"; do
        for hidden_dim in "${hidden_dims[@]}"; do
            for dataset in "${datasets[@]}"; do
                for fusion in "${fusion_models[@]}"; do
                    for model in "${model_names[@]}"; do
                        jobname="${model}_${fusion}_${dataset}_lr${lr}_dr${dr}"

                        increment="$first_run"
                        env | grep hidden_dim

                        # Export hyperparameters as environment variables
                        export jobname model fusion dataset lr dr increment hidden_dim

                        # Submit the job
                        bash "$jobfile"

                        first_run="false"
                    done
                done
            done
        done
    done
done

echo
echo "Done submitting many jobs!"