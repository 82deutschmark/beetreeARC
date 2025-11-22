# ARC-AGI

## Quickstart

```bash
git clone <repo-url>
cd ARC-AGI
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
cp config/api_keys.env.example config/api_keys.env  # then add your keys
# Export secrets (options below)
# Option A: export variables manually
export OPENAI_API_KEY=...
# Option B: source the config with auto-export in the shell session
set -a && source config/api_keys.env && set +a
python main.py data/first_100.json --reasoning none
```

The `.venv/` directory is ignored by Git, so every contributor can maintain their own environment locally without polluting the repo. Add any third-party libraries you install to `requirements.txt` so others can reproduce the environment quickly.

`main.py` accepts a JSON file containing a list of ARC task file paths (e.g., `data/first_100.json`). It loads each puzzle, packages the training examples plus each test input into an OpenAI prompt, requests a completion from `gpt-5.1`, and parses the returned grid. The predicted grid is compared against the ground-truth test output from the JSON file, and the script prints PASS/FAIL for each test case plus a summary table at the end. Use `--reasoning` to select the effort level (supported by `gpt-5.1`: `none`, `low`, `medium`, or `high`). Ensure `OPENAI_API_KEY` is exported (see example commands above) before running it; otherwise the script will exit with an error.

While running, stdout is limited to a streaming Markdown table with one row per test case. Copy the rows into `Results.md` (or another document) if you want to persist the results after the run.

An example task list (`data/first_100.json`) looks like:

```json
{
  "tasks": [
    "data/arc-agi-2-training/00576224.json",
    "data/arc-agi-2-training/007bbfb7.json",
    "... more task paths ..."
  ]
}
```
