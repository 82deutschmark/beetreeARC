# Grid Parsing Regression Test Suite

This directory contains test cases for the `parse_grid_from_text` function in `src/utils.py`. These cases are extracted from actual model execution logs to ensure robustness against various LLM output formats.

## Structure

*   `cases/`: Contains individual text files. Each file holds the "Full raw LLM response" from a model run.
*   `../test_grid_parsing.py`: The pytest suite that runs `parse_grid_from_text` against every file in `cases/`.

## Adding New Test Cases

If you encounter a log file where the grid extraction failed (or succeeded in a tricky way that you want to preserve), you can use the helper script in `tests/tools/` to extract it.

### Using `extract_cases_from_logs.py`

This script extracts the raw responses from a JSON log file and saves them as text files in the test case directory.

```bash
python3 tests/tools/extract_cases_from_logs.py <path_to_log_file.json>
```

(Note: You may need to modify the script slightly if you want to target specific runs or change the output directory, currently it defaults to `tests/grid_parsing_cases/cases` and might pick random files if run without arguments as per recent edits. Check the `if __name__ == "__main__":` block).

## Debugging

If a test case fails, you can use `tests/tools/debug_extraction.py` to run the parser against a specific log file and print detailed debug information (or just run pytest with `-s` to see stdout).

To inspect the content of a specific run in a large log file without extracting everything, use `tests/tools/inspect_log_entry.py`.
