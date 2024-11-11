#!/bin/bash -l

# Name of the sbatch script we want to submit many times
jobfile="jobs/ope_grid_search_payload.sh"

# Values for hyperparameters
env_vals=('LunarLander' 'CartPole' 'MountainCar')
# env_vals=('LunarLander')
ope_vals=('ips' 'dm_fqe' 'dm_fqe' 'dm_fqe' 'dm_fqe' 'dr_fqe' 'dr_fqe' 'dr_fqe' 'dr_fqe' 'ppo_ope' 'ppo_ope' 'ppo_ope' 'ppo_ope' 'ppo_ope' 'ppo_ope')
lr_vals=(0 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.00025 0.00025 0.0025 0.0025 0.00025 0.00025)
batch_size_vals=(0 2048 2048 2048 2048 2048 2048 2048 2048 2048 2048 2048 2048 2048 2048)
hidden_layer_neurons_vals=(0 64 64 64 128 64 64 64 128 64 64 64 64 128 128)
n_iters_vals=(0 50 25 75 50 50 25 75 50 0 0 0 0 0 0)
update_epochs_vals=(0 0 0 0 0 0 0 0 0 40 4 40 4 40 4)
clip_coef_vals=(0 0 0 0 0 0 0 0 0 0.2 0.2 0.2 0.2 0.2 0.2)
mini_batches_vals=(0 0 0 0 0 0 0 0 0 32 32 32 32 32 32)

echo
echo "Preparing to submit many jobs..."
echo

# For each value of experimental hyperparameter we want to test
for env in "${env_vals[@]}"; do
    for i in "${!ope_vals[@]}"; do
        jobname=$env'_'$ope
        op='output/'$env'_'$ope'.o'
        err='error/'$env'_'$ope'.e'
        ope=${ope_vals[i]}
        lr=${lr_vals[i]}
        batch_size=${batch_size_vals[i]}
        hidden_layer_neurons=${hidden_layer_neurons_vals[i]}
        n_iters=${n_iters_vals[i]}
        update_epochs=${update_epochs_vals[i]}
        clip_coef=${clip_coef_vals[i]}
        mini_batches=${mini_batches_vals[i]}
        
        # echo $ope' '$lr' '$batch_size' '$hidden_layer_neurons' '$n_iters' '$update_epochs' '$clip_coef' '$mini_batches


        # Export hyperparameters as environment variables
        export env ope lr batch_size hidden_layer_neurons n_iters update_epochs clip_coef mini_batches

        # Submit the job
        sbatch --job-name=$jobname --output=$op --error=$err $jobfile
    done
done

echo
echo "Done submitting many jobs!"