#!/bin/bash
#SBATCH --job-name=json_all
#SBATCH --output=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/json/logs/all_%j.out
#SBATCH --error=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/json/logs/all_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=140G
#SBATCH --gres=gpu:l40s:1

GENLM_ENV="/home/mila/b/benjamin.lebrun/miniconda3/envs/genlm"
PROJECT_ROOT="/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/json"

module load anaconda/3
conda activate $GENLM_ENV

RESULTS_PARENT_DIR="results-mt-350"
TASKS="Github_trivial Github_easy Github_medium Github_hard"
#"Glaiveai2K JsonSchemaStore Kubernetes WashingtonPost Snowplow Github_ultra"

sh $PROJECT_ROOT/jobs/twist.sh 15 $RESULTS_PARENT_DIR $TASKS
sh $PROJECT_ROOT/jobs/sr.sh 15 $RESULTS_PARENT_DIR $TASKS
#sh $PROJECT_ROOT/jobs/lm.sh 1 $RESULTS_PARENT_DIR $TASKS
#sh $PROJECT_ROOT/jobs/swar.sh 5 $RESULTS_PARENT_DIR $TASKS
#sh $PROJECT_ROOT/jobs/lcd.sh 1 $RESULTS_PARENT_DIR $TASKS
