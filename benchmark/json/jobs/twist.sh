#!/bin/bash
#SBATCH --job-name=json_twist
#SBATCH --output=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/json/logs/twist/json_twist_%j.out
#SBATCH --error=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/json/logs/twist/json_twist_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:l40s:1

N_PARTICLES=${1:-50}
OUTPUT_PARENT_DIR=${2:-"results"}
TASKS=${3:-"Github_trivial Github_easy Github_medium Github_hard"}

GENLM_ENV="/home/mila/b/benjamin.lebrun/miniconda3/envs/genlm"
PROJECT_ROOT="/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/json"

module load anaconda/3
conda activate $GENLM_ENV

ESS_THRESHOLD=$(echo "($N_PARTICLES - 0.5)/$N_PARTICLES" | bc -l)

python $PROJECT_ROOT/run_inference.py \
--model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
--output_dir $PROJECT_ROOT/${OUTPUT_PARENT_DIR}/twist-l40s-${ESS_THRESHOLD}-${N_PARTICLES} \
--n_particles $N_PARTICLES \
--lm_args '{"engine_opts" : {"max_model_len" : 10000, "dtype":"half"}}' \
--sampler_name lm \
--use_critic \
--time_sampler \
--verbosity 0 \
--ess_threshold $ESS_THRESHOLD \
--tasks $TASKS
