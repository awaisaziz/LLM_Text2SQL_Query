# DAIL-SQL

## Environment Setup

### 1. Launch the CoreNLP server

- Make sure Java is installed first
- Download [Stanford CoreNLP v3.9.2](https://stanfordnlp.github.io/CoreNLP/history.html), unzip it to the folder ./third_party, and then launch the CoreNLP server

```
cd third_party/stanford-corenlp-full-2018-10-05
nohup java -mx4g -cp "\*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer &
cd ../../
```

### 2. Make new environment

```
conda create -n dail-sql python=3.8
conda activate dail-sql
python -m pip install --upgrade pip
pip install -r requirements.txt
python nltk_downloader.py
```

### 3. Download the Spider dataset

Download the [Spider dataset](https://yale-lily.github.io/spider) and place it in the folder `./dataset/spider`

## Data Preprocess

```
python data_preprocess.py
```

## Prompt Generation

Select examples with masked question similarity:

```
python generate_question.py \
--data_type spider \
--split test \
--tokenizer gpt-3.5-turbo \
--max_seq_len 4096 \
--prompt_repr SQL \
--k_shot 9 \
--example_type QA \
--selector_type  EUCDISQUESTIONMASK
```

## Calling the LLM

Without voting:

```
python ask_llm.py \
--openai_api_key [your_openai_api_key]  \
--model gpt-3.5-turbo \
--question [prompt_dir]
```

## Evaluation of DAIL-SQL

- In this evaluation, GPT-3.5-Turbo was used as the LLM for SQL generation, and each query was generated without self-consistency voting
- Results are derived from the attached evaluation log (Eval_RESULTS_MODEL-gpt-3.5-turbo.txt)
  | Method | Dev EM | Dev EX | Test EM | Test EX |
  | --------- | --------- | --------- | --------- | --------- |
  | DAIL-SQL+GPT-3.5-Turbo | 0.52 | 0.68 | - | - |

## Acknowledgements

This implementation and evaluation are based on the official DAIL-SQL paper and open-source repository:

- Paper: “Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation,” Gao et al., 2023
- GitHub: https://github.com/BeachWang/DAIL-SQL
