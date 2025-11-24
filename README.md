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


## Analysis

### Base Model Performance
[Base Model Analysis](Results.md)

- Overall, similar performance across the smartest versions of OpenAI, Claude and Gemini
- GPT-5.1 with no reasoning is the only viable low latency and low cost model, though its performance is very low. Should only be used for very basic tasks.
- For a slightly smarter model it seems that Sonnet 4.5 with thinking turned off is a good choice. Higher latency and cost, but significant gain in performance. This could be the base solution model, whereas GPT-5.1 could be used as a "parsing model"
- Ideally, something should be stitched together using these models, and then the bigger models as "teachers" (extracting generic strategies, structuring the problem space, etc) and potentially also be used selectively as sanity checks on the very hardest problems
- GPT-5.1 with High reasoning performs the best (note: this is different from the official leaderboard). If perfectly combined with Gemini 3 High and Sonnet 4.5 with maximum thinking budget, around ~half of the errors are eliminated as compared to GPT 5.1 High stand-alone. E.g. there is significant non-overlap in the problems they solve. And, since Gemini 3 High and Sonnet 4.5 max budget are both lower latency, the only consideration in doing all three is cost - assuming the best solution can correctly be identified.

### Grid Format
[Grid Format Analysis](ANALYZE_GRID_FORMAT.md)

- 9 different basic formats were analyzed
- Formats that are not spoken-language-friendly clearly underperform (ascii, compact).
- Regular formats that have clear natural-language-separators perform roughly the same (xml, spaces, comma, semicolon, python) though there are some differences
- There is potential in formats that capture the nature of specific problems, like a sparse representation or a binary color representation, although the performance uplift is small enough that it may not be worth implementing it
- Specifically, sparse does outperform on certain sparse problems (e.g. https://arcprize.org/play?task=1cf80156). The uplift - if perfectly identified - is ~10% so it's not completely negligible, but it might be subsumed by more obvious improvements. Hence, will likely leave it for later
- CSV is the highest performing format, and outperforms the other natural-language-friendly formats by 10-20%. Python and space/newline-separated are the two runner ups but they underperform with 90% significance (sampled across 3x1000 problems)
- Formats not (yet) explored: object representation yet (e.g. islands, matched over rotations), condensed (e.g. 3 black, then 10 blue, ...), input-to-output diff, image/multimodal
- I have not yet explored the performance increase from having multiple representations present at the same time
- Conclusion: For now, will use CSV grid format only, and later on explore problem-specific representations (sparse, object, diff, image, etc)
