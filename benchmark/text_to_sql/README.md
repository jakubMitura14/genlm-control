1. Download the Spider dataset in the `data` directory
```bash
mkdir data
cd data
mkdir -p spider
cd spider
gdown 'https://drive.google.com/u/0/uc?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download'
unzip spider_data.zip
```

2. Install dependencies
```bash
cd ../..
pip install -r requirements.txt
```

3. Run inference
```bash
python run_inference.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --raw_spider_dir data/spider_data --output_dir results --n_particles 5 --max_tokens 100
```
