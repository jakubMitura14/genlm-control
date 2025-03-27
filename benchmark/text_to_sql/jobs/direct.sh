#!/bin/bash
#SBATCH --job-name=text_to_sql_direct
#SBATCH --output=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/text_to_sql/logs/text_to_sql_direct_%j.out
#SBATCH --error=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/text_to_sql/logs/text_to_sql_direct_%j.err
#SBATCH --time=18:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:40gb:1

GENLM_ENV="/home/mila/b/benjamin.lebrun/miniconda3/envs/genlm"
PROJECT_ROOT="/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/text_to_sql"

module load anaconda/3
conda activate $GENLM_ENV

python $PROJECT_ROOT/run_inference.py \
--model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
--raw_spider_dir $PROJECT_ROOT/data/spider_data \
--grammar_dir $PROJECT_ROOT/data/grammars \
--output_dir /home/mila/b/benjamin.lebrun/scratch/results/direct \
--n_particles 5 \
--max_tokens 100 \
--lm_args '{"engine_opts" : {"max_model_len" : 10000, "dtype":"half"}}' \
--sampler_args '{"K":100}' \
--overwrite \
--sampler_name top-k \
--time_sampler \
--verbosity 0 \
--ess_threshold 0.5 \
--timeout 900 \
--cache_clear_interval 10
