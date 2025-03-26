#!/bin/bash
#SBATCH --job-name=json_sr
#SBATCH --output=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/json/logs/sr/json_sr_%j.out
#SBATCH --error=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/json/logs/sr/json_sr_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:l40s:1

N_PARTICLES=${1:-45}
OUTPUT_PARENT_DIR=${2:-"results"}
TASKS=${3:-"Github_trivial Github_easy Github_medium Github_hard"}

GENLM_ENV="/home/mila/b/benjamin.lebrun/miniconda3/envs/genlm"
PROJECT_ROOT="/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/json"

module load anaconda/3
conda activate $GENLM_ENV

# Sampler rerank is IS (ess_threshold=0) with the LM as a proposal and a critic applied only at the end.

python $PROJECT_ROOT/run_inference.py \
--model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
--output_dir $PROJECT_ROOT/${OUTPUT_PARENT_DIR}/sr-l40s-${N_PARTICLES} \
--n_particles $N_PARTICLES \
--lm_args '{"engine_opts" : {"max_model_len" : 10000, "dtype":"half"}}' \
--sampler_name lm \
--use_critic \
--time_sampler \
--ess_threshold 0 \
--tasks $TASKS
