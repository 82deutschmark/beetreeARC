# Logs Schema Documentation

This document describes the schema of the JSON log files generated during the ARC-AGI solving process. The logs are stored in the `logs/` directory.

## Filename Convention

The files follow the pattern:
`{timestamp}_{task_id}_{test_index}_{step_name}.json`

Example: `2025-12-10_12-11-47_08ed6ac7_1_step_1.json`

- `timestamp`: Format `YYYY-MM-DD_HH-MM-SS`.
- `task_id`: The 8-character alphanumeric ARC task identifier.
- `test_index`: The index of the test pair being solved (usually 0 or 1).
- `step_name`: The step identifier (e.g., `step_1`, `step_5`, `step_finish`).

---

## Standard Step Files (`_step_1.json` to `_step_4.json`)

These files record the attempts made by various models at different stages. They follow a flat structure where each key at the root is a distinct model run.

### Root Object
| Key | Type | Description |
| :--- | :--- | :--- |
| `run_id` | Object | Details of a specific model's attempt. Key format: `{Model_Name}_{Step}_{Attempt}_{Timestamp}`. See [Model Attempt Object](#model-attempt-object). |
| `candidates_object` | Object | Aggregated unique solutions proposed by models. See [Candidates Object](#candidates-object). |
| `is_solved` | Boolean | **Step Success Indicator.** True if *at least one* candidate generated in this step matches the ground truth. |

### Model Attempt Object
Each model's attempt is stored under a unique `run_id` key (e.g., `claude-sonnet-4.5-no-thinking_1_step_1_1765394054.883797`).

| Key | Type | Description |
| :--- | :--- | :--- |
| `duration_seconds` | Number | Time taken for the LLM call. |
| `total_cost` | Number | Cost of the API call. |
| `input_tokens` | Integer | Number of tokens in the input prompt. |
| `output_tokens` | Integer | Number of tokens in the generated response. |
| `cached_tokens` | Integer | Number of cached tokens used (if any). |
| `Full raw LLM call` | String | The exact prompt sent to the LLM. |
| `Full raw LLM response` | String | The raw text response received from the LLM. |
| `Extracted grid` | Array (2D) | The 9x9 grid extracted from the LLM's response. |
| `is_correct` | Boolean | **Specific Success Indicator.** True if *this specific* extracted grid matches the ground truth. |

### Candidates Object
This object aggregates identical grids proposed by different models. Keys are string representations of the grid tuples.

| Key | Type | Description |
| :--- | :--- | :--- |
| `grid` | Array (2D) | The actual grid structure. |
| `count` | Integer | Number of models that proposed this specific grid. |
| `models` | Array of Strings | List of model identifiers that proposed this grid. |
| `is_correct` | Boolean | **Specific Success Indicator.** True if *this specific* unique grid candidate matches the ground truth. |

---

## Complex Step File (`_step_5.json`)

Step 5 involves more complex pipelines. Unlike Steps 1-3, this file is **hierarchical**. The root keys represent **strategies**, and each strategy object contains multiple model runs or sub-components.

### Root Object (Strategies)
| Key | Type | Description |
| :--- | :--- | :--- |
| `trigger-deep-thinking` | Object | Strategy using "deep thinking" protocols. Contains a dictionary of [Model Attempt Objects](#model-attempt-object) keyed by `run_id`. |
| `image` | Object | Strategy using visual representations. Contains a dictionary of [Model Attempt Objects](#model-attempt-object) keyed by `run_id`. |
| `generate-hint` | Object | Strategy where models generate hints before solving. Contains runs and metadata. |
| `objects_pipeline` | Object | Strategy using object-based reasoning pipelines. Contains variant runs. |
| `candidates_object` | Object | Aggregated solutions from all Step 5 strategies. |
| `is_solved` | Boolean | **Step Success Indicator.** True if *at least one* candidate generated in this step matches the ground truth. |

### Strategy-Specific Structures

#### `generate-hint`
Contains:
1.  **Model Runs**: Standard [Model Attempt Objects](#model-attempt-object) keyed by `run_id`.
2.  **`hint_generation`**: A sub-object detailing the specific hint creation process.
    *   `model`: The model used for generating the hint.
    *   `Full raw LLM call`: Prompt used to generate the hint.
    *   `Full raw LLM response`: Raw response containing the hint.
    *   `Extracted hint`: The parsed hint string.
    *   Usage stats (`duration_seconds`, `total_cost`, etc.).

#### `objects_pipeline`
Contains different variants of the object pipeline (e.g., `gemini-3-low...`, `gpt-5.1...`) keyed by `run_id`.
Depending on the variant, it may contain specific pipeline traces such as:
*   **`gemini_gen`** or **`opus_gen`**: Detailed breakdown of the object reasoning pipeline.
    *   `extraction`: Object identification step (model, prompt, response, summary).
    *   `transformation`: Transformation rule identification step (model, prompt, response, summary).
    *   `solution_prompt`: The final prompt sent to the solver model using the extracted info.

---

## Finish File (`_step_finish.json`)

This file contains the definitive results and the logic used to select the final solution.

### Root Object
| Key | Type | Description |
| :--- | :--- | :--- |
| `candidates_object` | Object | Final aggregation of all proposed solutions from all steps. |
| `selection_details` | Object | **Definitive location** for the selection logic metadata. See [Selection Details](#selection-details). |
| `picked_solutions` | Array of Objects | The solutions selected by the agent as the best candidates. |
| `correct_solution` | Array (2D) | The ground truth grid. |
| `result` | String | **Run Verdict.** The final result of the entire task run. Values: "PASS" or "FAIL". |

### Selection Details
This object details how the final solution was picked from the candidates.

| Key | Type | Description |
| :--- | :--- | :--- |
| `judges` | Object | Contains `logic` and `consistency` judges. |
| `selection_process` | Object | Summary of votes and the decision process (e.g., Consensus vs. Auditor). |

#### Judges Object
Contains `logic` and `consistency` keys. Each is an object with:
- `model`: The model used for judging.
- `prompt`: The prompt sent to the judge.
- `response`: The judge's raw response.
- `parsed`: Structured evaluation (e.g., candidate rankings, tiers).
- Usage stats (`duration_seconds`, etc.).

### Picked Solution Object
| Key | Type | Description |
| :--- | :--- | :--- |
| `grid` | Array (2D) | The proposed grid. |
| `count` | Integer | Number of votes/models supporting this grid. |
| `models` | Array of Strings | Models that proposed this grid. |
| `is_correct` | Boolean | **Specific Success Indicator.** True if *this specific* picked solution matches the correct solution. |