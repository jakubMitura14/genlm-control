Make sure `genlm-control` is installed (see the instructions in the library README).

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Download the Spider dataset in the `data` directory
```bash
cd data
gdown 'https://drive.google.com/u/0/uc?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download'
unzip spider_data.zip
```

3. Run inference

From the `text_to_sql` directory, run inference with `run_inference.py`.

For example:

```bash
python run_inference.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --raw_spider_dir data/spider_data \
    --output_dir results/test \
    --n_particles 5 \
    --max_tokens 100 \
    --lm_args ' {"engine_opts" : {"max_model_len" : 10000}}'
```

4. Run evaluation
```bash
python run_evaluation.py \
    --results_dir results/test \
    --n_workers 5 \
    --raw_spider_dir data/spider_data \
    --output_pkl results/test/evaluation_results.pkl
```
