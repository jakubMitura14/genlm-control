# Text-to-SQL

This directory contains the code for running inference on the Spider dataset.

# 1. Install dependencies

Make sure `genlm-control` is installed (see the instructions in the library README).

Install the dependencies for this evaluation in the `requirements.txt` file with the following command:
```bash
pip install -r requirements.txt
```

Download the required `nltk` data:
```bash
python -m nltk.downloader punkt_tab
```


# 2. Download the Spider dataset in the `data` directory

Download and unzip the Spider dataset in the `data` directory with the following command:
```bash
cd data
gdown 'https://drive.google.com/u/0/uc?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download'
unzip spider_data.zip
```

# 3. Run inference

From the `text_to_sql` directory, run inference with `run_inference.py`.

For example:

```bash
python run_inference.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --raw_spider_dir data/spider_data --output_dir results/test --n_particles 10 --max_tokens 100 --lm_args '{"engine_opts" : {"max_model_len" : 10000}}'
```

See the `run_inference.py` file for more details on the arguments.

# 4. Run evaluation

From the `text_to_sql` directory, run the evaluation pipeline with `run_evaluation.py`. This actually evaluates the results of inference.

For example:

```bash
python run_evaluation.py \
    --results_dir results/test \
    --n_workers 5 \
    --raw_spider_dir data/spider_data \
    --output_pkl results/test/evaluation_results.pkl
```

Or run the evaluation for all the results in the `results` directory with the following command:

```bash
./eval_all.sh results/*
```
