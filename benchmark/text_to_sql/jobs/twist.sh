#!/bin/bash
#SBATCH --job-name=sql_twist
#SBATCH --output=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/text_to_sql/logs/twist/sql_twist_%j.out
#SBATCH --error=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/text_to_sql/logs/twist/sql_twist_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=140G
#SBATCH --gres=gpu:l40s:1

N_PARTICLES=${1:-10}
OUTPUT_PARENT_DIR=${2:-"results"}
MODEL_NAME=${3:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}

GENLM_ENV="/home/mila/b/benjamin.lebrun/miniconda3/envs/genlm"
PROJECT_ROOT="/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/text_to_sql"

module load anaconda/3
conda activate $GENLM_ENV

MAX_TOKENS=100
ESS_THRESHOLD=$(echo "($N_PARTICLES - 0.5)/$N_PARTICLES" | bc -l)

python $PROJECT_ROOT/run_inference.py \
--model_name $MODEL_NAME \
--raw_spider_dir $PROJECT_ROOT/data/spider_data \
--grammar_dir $PROJECT_ROOT/data/grammars \
--output_dir $PROJECT_ROOT/${OUTPUT_PARENT_DIR}/twist-l40s-${ESS_THRESHOLD}-${N_PARTICLES} \
--n_particles $N_PARTICLES \
--max_tokens $MAX_TOKENS \
--lm_args '{"engine_opts" : {"max_model_len" : 10000, "dtype":"half"}}' \
--sampler_name lm \
--use_critic \
--time_sampler \
--verbosity 0 \
--ess_threshold $ESS_THRESHOLD
