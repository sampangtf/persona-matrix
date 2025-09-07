# Experiment Data

Artifacts produced when generating the controlled dimension-shifted dataset and evaluating summaries.

## PersonaMatrix Evaluator

### `cached_criteria/`
JSON files caching rubric definitions used by evaluator models.
Schema:
- `task_key` – identifier of task and model.
- `timestamp` – ISO-8601 generation time.
- `criteria` – array of objects with:
  - `name`, `description`
  - `sub_criteria` – array with:
    - `name`, `description`
    - `accepted_values` – array of strings.

### `evaluation_results/`
JSONL files; each line stores evaluation for one case across all summary variants.
Top-level fields per line:
- `case_id`, `case_name`, `summary`
- `evaluation_results` – object keyed by `Level_0_Original`,
  `Depth_Level_1`–`Depth_Level_4`, `Precision_Level_1`–`Precision_Level_4`,
  `Procedural_Level_1`–`Procedural_Level_4`. Each entry contains:
  - `case_id`, `case_name`, `summary_key`, `summary_length`
  - `baseline` – {`framework`, `status`, `parsed`, `error`, `Performance`, `Score`}
  - `agent_eval` – {`framework`, `status`, `criteria_count`, `result`}
  - `persona_eval` – {`framework`, `status`, `persona_count`, `result`}
  - `errors` – list of strings.
Within `persona_eval.result`, persona names map to
`evaluations` objects, each containing criterion →
{`performance`, `numerical_score`} pairs.

## Controlled Shifted Dimensions Dataset

### `shifted_dataset/`
JSON files of dimension-shifted summaries.
Structure:
- `metadata` – generation info (`generation_timestamp`, `generator_version`, `model`, `total_cases`, `dimensions`).
- `cases` – array of case objects:
  - `case_id`, `generation_metadata`, `original_case_data`
  - `dimension_variations` – mapping for each dimension (`Depth`, `Precision`, `Procedural`) with:
    - `dimension_name`, `dimension_description`
    - `levels` – keys `"0"`–`"4"`; each level holds:
      - `transformed_text`
      - `metadata` (word counts, citation/procedure stats, readability metrics)
      - `validation_result` (`passed`, `hard_checks`, `llm_critique`, `new_metadata`, `failure_reason`)
      - `attempts`, `success`.
