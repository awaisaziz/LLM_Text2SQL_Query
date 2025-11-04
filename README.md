# Text-to-SQL Baseline with OpenRouter

This repository contains a lightweight baseline pipeline for evaluating large language models on the [Spider](https://yale-lily.github.io/spider) Text-to-SQL benchmark using the OpenRouter API. The goal is to provide a clean, modular starting point that can be easily extended with few-shot prompting, schema formatting improvements, and caching.

## Repository structure

```
root
├──text2sql/
├──├── app.py              # Main entry point for running inference
├──├── config.json         # Dataset and model configuration
├──├── data_utils.py       # Spider dataset loading helpers
├──├── evaluate.py         # Wrapper around the official Spider evaluation script
├──├── llm.py              # OpenRouter API client
└──├── prompt_template.py  # Prompt construction utilities
├── outputs/            # Store the predictions.sql file predicted sql query from the LLM
├── logs/               # Default directory for log output
├── README.md           # This file
├── .venv/              # Local Python virtual environment
├── requirements.txt    # Python dependencies
├── evaluation.py       # Official evaluation file from the Spider Github repo
├── process_sql.py      # Official process_sql.py file from the Spider Github repo
└── install.py          # Once install the package nltk module
```

The Spider dataset should be available locally under `./spider_data/` with the following expected files:

- `dev.json`
- `tables.json`
- `dev_gold.sql`
- `database/` (directory containing the SQLite databases)
- `evaluate.py` (official Spider evaluation script)

Update `config.json` if your dataset lives elsewhere.

## Environment setup

Create and activate a Python virtual environment:

```bash
python -m venv text2sql/.venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r text2sql/requirements.txt
```

Copy the example environment file and add your API key:

```bash
cp text2sql/.env.example text2sql/.env
```

## Running inference

The baseline is invoked via `app.py`. A minimal example that runs the first 20 development examples using the DeepSeek open model is shown below:

```bash
python -m text2sql.app --model deepseek/deepseek-r1:free --num_samples 20 --out outputs/predictions.jsonl
```

```bash
python DIN-SQL.py --dataset spider_data/ --output outputs/predicted_sql.txt
```

The resulting JSONL file contains four fields per line: `question`, `gold_sql`, `pred_sql`, and `db_id`.

## Evaluation

Use `evaluate.py` to call the official Spider evaluation script and obtain exact match and execution accuracy metrics:

```bash
python install.py
```

```bash
python evaluation.py
--gold spider_data/dev_gold.sql
--pred outputs/predictions.sql
--db spider_data/database
--table spider_data/tables.json
--etype all
```

The script will create a temporary `.sql` file, run `spider_data/evaluate.py`, and print the reported metrics.

## Extending the baseline

- Modify `prompt_template.py` to add few-shot demonstrations, schema reformatting, or additional instructions.
- Enhance `llm.py` to capture latency, prompt/response token usage, or to integrate caching.
- Add experiments and ablation studies under a new module without touching the core baseline files.

Contributions are welcome!
