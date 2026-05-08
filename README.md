# Code for the paper: DynaCall: Dynamic Function Calling with Branching, Replanning, and Semantic Mapping

## Installation

1. Create a Python environment.

```sh
conda create --name dynacall python=3.10 -y
conda activate dynacall
```

2. Install dependencies from the project root.

```sh
pip install -r ../requirements.txt
```

## Runtime setup

Set your OpenAI-compatible endpoint before running:

```sh
export API_KEY="sk-xxx"
export BASE_URL="https://.../v1"
```

Some benchmarks also need extra credentials:

- `bfcl_ws`: `export LANGSEARCH_API_KEY="xxx"`

## Main entry point

Run experiments with:

```sh
python run_dynacall.py --benchmark_name {benchmark} --store {output_json}
```

Supported benchmarks in this code release:

- `gaia`
- `bfcl_ws`
- `parallelqa`
- `movie`

Common options:

- `--model_name`: override the default model
- `--row_number`: run one row, a comma list, or a range such as `1`, `1,3,8`, or `1-10`
- `--logging`: print detailed execution logs
- `--max_questions`: concurrency across questions
- `--use_early_execution`: enable streaming planning/execution
- `--use_function_coalescing`: enable function coalescing
- `--cache_file`: persist cached tool results locally
- `--gaia_dataset_path`: override the GAIA json/jsonl file
- `--gaia_files_root`: override the GAIA attachment root

## Examples

### GAIA

```sh
python run_dynacall.py \
  --benchmark_name gaia \
  --store ./tmp/gaia.json \
  --model_name gpt-5.4 \
  --logging
```

### BFCL-WS

```sh
python run_dynacall.py \
  --benchmark_name bfcl_ws \
  --store ./tmp/bfcl_ws.json \
  --model_name gpt-5.4 \
  --logging
```

### ParallelQA

```sh
python run_dynacall.py \
  --benchmark_name parallelqa \
  --store ./tmp/parallelqa.json \
  --model_name gpt-5.4 \
  --logging
```

### MovieRec

```sh
python run_dynacall.py \
  --benchmark_name movie \
  --store ./tmp/movie.json \
  --model_name gpt-5.4 \
  --logging
```
