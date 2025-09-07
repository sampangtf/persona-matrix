# PersonaMatrix: Persona-Aware Evaluation of Legal Summarization

This repository contains the code and data that accompany the paper **"PersonaMatrix: A Recipe for Persona-Aware Evaluation of Legal Summarization."** It provides tools for generating controlled, dimension-shifted legal case summaries and evaluating them through both generic and persona-specific lenses.

## Repository Structure

### `dataset-generator/`
A LangGraph workflow that converts Civil Rights Clearinghouse case summaries into variants along three quality dimensions using an **Extractor → Transformer → Validator** agent pipeline. Each run can shift depth, technical precision, or procedural detail across five levels to create progressive summary variants.

### `personamatrix-evaluator/`
Evaluation framework supporting both **AgentEval** (generic criteria) and **PersonaEval** (multi-persona criteria). It includes utilities for generating rubrics, caching criteria, and scoring summaries with different large language models.

### `experiment/`
Artifacts produced during dataset generation and evaluation. Contains cached rubric definitions, evaluation outputs for each summary variant, and the resulting controlled, dimension-shifted dataset.

## Getting Started

Each subdirectory includes detailed documentation and scripts for reproducing the dataset and evaluation results. Please refer to their respective `README.md` files for installation and usage instructions.

## License

This project is released under the terms of the MIT License. See [LICENSE](LICENSE) for details.
