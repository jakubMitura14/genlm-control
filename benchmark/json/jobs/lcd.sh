#!/bin/bash
#SBATCH --job-name=json_lcd
#SBATCH --output=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/json/logs/lcd/json_lcd_%j.out
#SBATCH --error=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/json/logs/lcd/json_lcd_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:l40s:1

N_PARTICLES=${1:-1}
OUTPUT_PARENT_DIR=${2:-"results"}
TASKS=${3:-"Github_trivial Github_easy Github_medium Github_hard"}

GENLM_ENV="/home/mila/b/benjamin.lebrun/miniconda3/envs/genlm"
PROJECT_ROOT="/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/json"

module load anaconda/3
conda activate $GENLM_ENV

# This is adapative rejection sampling without proper weights.
# Only one run of the sampler.

python $PROJECT_ROOT/run_inference.py \
--model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
--output_dir $PROJECT_ROOT/${OUTPUT_PARENT_DIR}/lcd-l40s-${N_PARTICLES} \
--n_particles $N_PARTICLES \
--lm_args '{"engine_opts" : {"max_model_len" : 10000, "dtype":"half"}}' \
--sampler_name swar \
--sampler_args '{"proper_weights" : false}' \
--time_sampler \
--verbosity 0 \
--tasks $TASKS \
--ess_threshold 0
