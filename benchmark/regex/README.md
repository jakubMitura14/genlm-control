# Regex Benchmark

Install dependencies with:

```
pip install -r requirements.txt
```

Run inference with:

```
python run_inference.py  --output_dir results/text --n_particles 5 --sampler_name swar --verbosity 1 --lm_args '{"engine_opts" : {"dtype" : "half", "max_model_len" : 10000}}'
```

Evaluation will happen as part of the inference script. The evaluation results will also be saved in the output directory.

Results will be saved in the output directory in the following format:

```
{instance_id}_result.pkl
```

You can then analyze the results with:

```
python analyze_results.py --dirs results/*  --n_bootstrap 10000 --confidence 0.95
```

This will save the results to `analysis_results.csv`.
