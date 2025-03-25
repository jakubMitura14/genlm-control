#!/bin/bash
#SBATCH --job-name=regex_lm
#SBATCH --output=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/regex/logs/lm/regex_lm_%j.out
#SBATCH --error=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/regex/logs/lm/regex_lm_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=140G
#SBATCH --gres=gpu:l40s:1

N_PARTICLES=${1:-1}
OUTPUT_PARENT_DIR=${2:-"results-ct"}

GENLM_ENV="/home/mila/b/benjamin.lebrun/miniconda3/envs/genlm"
PROJECT_ROOT="/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/regex"

module load anaconda/3
conda activate $GENLM_ENV

MAX_TOKENS=32

python $PROJECT_ROOT/run_inference.py \
--model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
--output_dir $PROJECT_ROOT/${OUTPUT_PARENT_DIR}/lm-l40s-${N_PARTICLES} \
--n_particles $N_PARTICLES \
--max_tokens $MAX_TOKENS \
--lm_args '{"engine_opts" : {"max_model_len" : 10000, "dtype":"half"}}' \
--sampler_name lm \
--time_sampler \
--verbosity 0 \
--ess_threshold 0
