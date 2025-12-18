# Text-to-SQL Baseline

This repository contains a lightweight baseline pipeline for evaluating large language models on the [Spider](https://yale-lily.github.io/spider) Text-to-SQL benchmark using OpenAI-compatible providers. The goal is to provide a clean, modular starting point that can be easily extended with few-shot prompting, schema formatting improvements, and caching.

## Repository structure

```
root
├── text2sql/
│   ├── main.py                 # CLI entry point
│   ├── config/                 # Configuration loader and defaults
│   ├── generation/             # SQL generation pipeline
│   ├── models/                 # Router definitions
│   │   └── provider/           # Provider-specific router settings (deepseek, chatgpt, openrouter)
│   ├── prompt/                 # Prompt engineering utilities (zero-shot)
│   └── util/                   # Dataset loader, SQL cleaner, logging helpers
├── output/                     # Run artifacts
│   ├── log/                    # Log files
│   └── predicted/              # Generated SQL predictions
├── spider_data/                # Spider dataset root (dev.json, tables.json, database/)
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── evaluation.py               # Official evaluation file from the Spider Github repo
├── process_sql.py              # Official process_sql.py file from the Spider Github repo
└── install.py                  # Once install the package nltk module
```

The Spider dataset should be available locally under `./spider_data/` with the following expected files:

- `dev.json`
- `tables.json`
- `dev_gold.sql`
- `database/` (directory containing the SQLite databases)
- `evaluate.py` (official Spider evaluation script)

Update `text2sql/config/config.json` if your dataset lives elsewhere.

## Environment setup

Create and activate a Python virtual environment:

```bash
python -m venv text2sql/.venv
.venv\Scripts\activate
```

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Set the appropriate environment variables for your chosen provider (for example `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, or `OPENAI_API_KEY`). A `.env` file placed alongside `text2sql/config/config.json` will also be loaded automatically.

## Running inference

The pipeline is invoked via `text2sql/main.py`. A minimal example that runs the first 20 development examples using the DeepSeek provider is shown below:

```bash
python -m text2sql.main --provider deepseek --model deepseek-chat --num_samples 20 --out predicted/deepseek_chat_predicted.sql
```

The resulting file contains one SQL query per line. Paths provided via `--out` are resolved under the `output/` directory unless an absolute path is given.

### Configuration

Default values live in `text2sql/config/config.json` and are loaded via `text2sql.config`:

```json
{
  "dataset_path": "./spider_data/",
  "default_provider": "deepseek",
  "default_model": "deepseek-chat",
  "num_sample": 100,
  "max_tokens": 8000,
  "request_delay": 0.0,
  "mode": "zero_shot",
  "db_root": "spider_data/database",
  "output_llm": "predicted/deepseek_chat_predicted.json",
  "tables_filename": "tables.json",
  "rag": {
    "num_retrieve": 200,
    "k": 4,
    "n": 5,
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "retrieval_examples_filename": "test.json",
    "retrieval_tables_filename": "test_tables.json"
  }
}
```

`dataset_path` should point to the folder containing `dev.json` and `tables.json`, while `output_llm` controls the default prediction filename (stored under `output/`). Retrieval pulls similar examples from `test.json`/`test_tables.json` while generation iterates through `dev.json`/`tables.json`.

All dataset, model, and RAG parameters are read from this JSON file. Command-line arguments can override selected values at runtime: `--provider`, `--model`, `--num_samples`, `--out`, `--mode`, `--k`, `--n`, and `--num_retrieve`.

## Retrieval-augmented Text-to-SQL (inference-only)

The repository also ships an inference-only, retrieval-augmented pipeline that layers cosine-similarity retrieval, self-consistency, and execution-based majority voting. All retrieval settings live in `text2sql/config/config.json` under the `rag` key. Run the pipeline over the development set directly from the main entry point (questions are read from `dev.json`):

```bash
python -m text2sql.main \
  --provider deepseek \
  --model deepseek-chat \
  --num_samples 2 \
  --out predicted/deepseek_chat_predicted.sql \
  --mode cot \
  --k 3 \
  --n 5 \
  --num_retrieve 200
```

This command will, for each selected example from `dev.json`, retrieve the top-`k` similar questions (and SQL) from `test.json`, build the prompt with the selected `mode` (e.g., `cot` for chain-of-thought), generate `n` SQL candidates, execute them, and return the execution-voted SQL. Provider/model defaults come from `default_provider` and `default_model`. The pipeline uses sentence-transformers for embeddings, scikit-learn for cosine similarity, SQLite for execution, and the configured chat provider for generation.

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
