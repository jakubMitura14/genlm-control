#!/bin/bash
#SBATCH --job-name=text_to_sql_all
#SBATCH --output=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/text_to_sql/logs/all_%j.out
#SBATCH --error=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/text_to_sql/logs/all_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=250G
#SBATCH --gres=gpu:l40s:1

GENLM_ENV="/home/mila/b/benjamin.lebrun/miniconda3/envs/genlm"
PROJECT_ROOT="/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/text_to_sql"

module load anaconda/3
conda activate $GENLM_ENV

RESULTS_PARENT_DIR="results-full"

MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"

sh $PROJECT_ROOT/jobs/swar.sh 5 $RESULTS_PARENT_DIR $MODEL_NAME
sh $PROJECT_ROOT/jobs/twist.sh 10 $RESULTS_PARENT_DIR $MODEL_NAME
sh $PROJECT_ROOT/jobs/sr.sh 10 $RESULTS_PARENT_DIR $MODEL_NAME

sh $PROJECT_ROOT/jobs/lm.sh 1 $RESULTS_PARENT_DIR $MODEL_NAME
sh $PROJECT_ROOT/jobs/lcd.sh 1 $RESULTS_PARENT_DIR $MODEL_NAME

sh $PROJECT_ROOT/jobs/swar.sh 10 $RESULTS_PARENT_DIR $MODEL_NAME
sh $PROJECT_ROOT/jobs/twist.sh 20 $RESULTS_PARENT_DIR $MODEL_NAME

sh $PROJECT_ROOT/jobs/swar.sh 20 $RESULTS_PARENT_DIR $MODEL_NAME
sh $PROJECT_ROOT/jobs/twist.sh 40 $RESULTS_PARENT_DIR $MODEL_NAME

MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"

sh $PROJECT_ROOT/jobs/swar.sh 5 $RESULTS_PARENT_DIR $MODEL_NAME
sh $PROJECT_ROOT/jobs/twist.sh 10 $RESULTS_PARENT_DIR $MODEL_NAME
sh $PROJECT_ROOT/jobs/sr.sh 10 $RESULTS_PARENT_DIR $MODEL_NAME

sh $PROJECT_ROOT/jobs/lm.sh 1 $RESULTS_PARENT_DIR $MODEL_NAME
sh $PROJECT_ROOT/jobs/lcd.sh 1 $RESULTS_PARENT_DIR $MODEL_NAME

sh $PROJECT_ROOT/jobs/swar.sh 10 $RESULTS_PARENT_DIR $MODEL_NAME
sh $PROJECT_ROOT/jobs/twist.sh 20 $RESULTS_PARENT_DIR $MODEL_NAME

sh $PROJECT_ROOT/jobs/swar.sh 20 $RESULTS_PARENT_DIR $MODEL_NAME
sh $PROJECT_ROOT/jobs/twist.sh 40 $RESULTS_PARENT_DIR $MODEL_NAME
