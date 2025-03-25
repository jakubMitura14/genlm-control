#!/bin/bash
#SBATCH --job-name=regex_swar
#SBATCH --output=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/regex/logs/swar/regex_swar_%j.out
#SBATCH --error=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/regex/logs/swar/regex_swar_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=140G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:4

GENLM_ENV="/home/mila/b/benjamin.lebrun/miniconda3/envs/genlm"
PROJECT_ROOT="/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/regex"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=1

module load anaconda/3
conda activate $GENLM_ENV

N_PARTICLES=5
MAX_TOKENS=32

python $PROJECT_ROOT/run_inference.py \
--model_name meta-llama/Llama-3.3-70B-Instruct \
--output_dir $PROJECT_ROOT/results/swar-l40s-70b-${N_PARTICLES} \
--n_particles $N_PARTICLES \
--max_tokens $MAX_TOKENS \
--lm_args '{"engine_opts" : {"max_model_len" : 2000, "tensor_parallel_size" : 4}}' \
--sampler_name swar \
--time_sampler \
--verbosity 0 \
--ess_threshold 0.5
