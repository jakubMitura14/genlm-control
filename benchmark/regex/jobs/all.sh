#!/bin/bash
#SBATCH --job-name=regex_all
#SBATCH --output=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/regex/logs/all_%j.out
#SBATCH --error=/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/regex/logs/all_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=140G
#SBATCH --gres=gpu:l40s:1

GENLM_ENV="/home/mila/b/benjamin.lebrun/miniconda3/envs/genlm"
PROJECT_ROOT="/home/mila/b/benjamin.lebrun/genlm/genlm-control/benchmark/regex"

module load anaconda/3
conda activate $GENLM_ENV

RESULTS_PARENT_DIR="results-ct2"

#sh $PROJECT_ROOT/jobs/lm.sh 1 $RESULTS_PARENT_DIR
sh $PROJECT_ROOT/jobs/swar.sh 5 $RESULTS_PARENT_DIR
sh $PROJECT_ROOT/jobs/twist.sh 50 $RESULTS_PARENT_DIR
sh $PROJECT_ROOT/jobs/sr.sh 40 $RESULTS_PARENT_DIR
sh $PROJECT_ROOT/jobs/lcd.sh 1 $RESULTS_PARENT_DIR
sh $PROJECT_ROOT/jobs/direct.sh 1 $RESULTS_PARENT_DIR
