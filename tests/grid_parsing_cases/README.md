# Grid Parsing Regression Test Suite

This directory contains test cases for the `parse_grid_from_text` function in `src/utils.py`. These cases are extracted from actual model execution logs to ensure robustness against various LLM output formats.

## Structure

*   `cases/`: Contains individual text files. Each file holds the "Full raw LLM response" from a model run. For each `.txt` file, there is a corresponding `.truth.json` file containing the expected grid (extracted by `claude-opus-4.5` in "no thinking" or "thinking" mode).
*   `../test_grid_parsing.py`: The pytest suite that runs `parse_grid_from_text` against every file in `cases/` and verifies it matches the ground truth.

## Running Tests

To run the full regression suite:

```bash
./.venv/bin/pytest tests/test_grid_parsing.py
```

## Known Failures

As of Dec 8, 2025, 107/109 tests pass. The following 2 cases are known to fail:

1.  **`gpt-5.1-high_2_step_1_1765155396.197504.txt`**
    *   **Issue:** Ground Truth Mismatch.
    *   **Details:** The raw text contains a row `1,1,1,1,3...` (4 ones). The `parse_grid_from_text` correctly extracts 4 ones. However, the Ground Truth LLM (`claude-opus`) consistently hallucinates/corrects this to `1,1,1,1,1,3...` (5 ones).
    *   **Verdict:** The parser is correct; the ground truth is flawed but preserved to document LLM disagreement.

2.  **`objects_pipeline_gemini-3-high_12_step_5_opus_gen_sol_1765142275.8652031.txt`**
    *   **Issue:** Ragged Grid (Width Mismatch).
    *   **Details:** The model output contained rows of varying lengths (mostly 25, some 26). The parser faithfully extracted this ragged grid (height 3). The test suite sanity check enforces strict rectangularity (`assert len(row) == width`), causing an assertion error (`26 == 25`).
    *   **Verdict:** The parser is faithfully extracting the model's malformed output. The failure is in the strictness of the test sanity check vs. the reality of model output quality.

## Adding New Test Cases

If you encounter a log file where the grid extraction failed, you can use the helper script in `tests/tools/` to extract it.

### Using `extract_cases_from_logs.py`

This script extracts the raw responses from a JSON log file and saves them as text files in the test case directory.

```bash
python3 tests/tools/extract_cases_from_logs.py <path_to_log_file.json>
```

## Generating Ground Truth

To generate `.truth.json` files for new cases using Claude Opus:

```bash
python3 tests/tools/generate_ground_truth.py
```

This requires `ANTHROPIC_API_KEY` in `config/api_keys.env`. It skips existing truth files unless `--force` is used.