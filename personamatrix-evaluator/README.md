# PersonaMatrix Evaluator

A sophisticated evaluation framework that provides both generic (AgentEval) and persona-specific (PersonaEval) evaluation capabilities for AI outputs, with specialized support for evaluating legal document summaries and other domain-specific content.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [API Reference](#api-reference)

## Overview

PersonaMatrix Evaluator provides two main evaluation frameworks:

1. **AgentEval**: Task-based evaluation framework that generates generic evaluation criteria
2. **PersonaEval**: Multi-persona evaluation framework that generates persona-specific criteria from different user perspectives

### Key Features

- **Dual Evaluation Modes**: Both generic and persona-specific evaluation capabilities
- **Criteria Caching**: Save and reuse evaluation criteria across sessions
- **LangGraph Workflows**: Orchestrated agent interactions using state graphs
- **Flexible LLM Support**: Compatible with various language models (OpenAI, Google Gemini, etc.)
- **Rate Limiting**: Built-in API rate limiting and management

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Create .env file
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Quick Start

The PersonaMatrix Evaluator supports two main workflows:

### 1. Generate Criteria

Generate evaluation criteria for a specific task using both AgentEval and PersonaEval frameworks:

```python
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Add the package to your path
sys.path.append('../personamatrix-evaluator/src')

# Import the frameworks
from src.core.evaluator import LangGraphAgentEval
from src.core.persona_evaluator import PersonaEvaluator
from src.models.state import Task
from src.models.persona import get_all_personas, get_persona_names

load_dotenv()

# Set up your LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.8)

# Define your evaluation task
task = Task(
    name="Legal Case Summary Utility Evaluation",
    description="""Evaluate utility of legal case summaries for its target user types. 
    Emphasize on whether the summaries is useful to end users, considering the context 
    of the usage, and background of users.""",
    expected_output="Comprehensive evaluation criteria for legal case summaries"
)

# Generate AgentEval (generic) criteria
agent_evaluator = LangGraphAgentEval(llm=llm)
agent_criteria = agent_evaluator.generate_criteria(
    task=task,
    additional_instructions="",
    max_round=2,
    use_subcritic=True
)

# Save AgentEval criteria to cache
agent_evaluator.save_criteria_to_cache(
    task_key="legal_summary_eval_gpt4o",
    criteria=agent_criteria,
    cache_file="cached_criteria/agent_eval_criteria_gpt4o.json"
)

# Generate PersonaEval (persona-specific) criteria
persona_evaluator = PersonaEvaluator(llm=llm)
all_personas = get_all_personas()
persona_names = get_persona_names()
evaluation_personas = [all_personas[name] for name in persona_names if name in all_personas]

persona_criteria = persona_evaluator.generate_persona_criteria(
    task=task,
    personas=evaluation_personas,
    additional_instructions="",
    max_round=2,
    use_subcritic=True
)

# Save PersonaEval criteria to cache
persona_evaluator.save_persona_criteria_to_cache(
    task_key="legal_summary_persona_eval_gpt4o",
    persona_criteria=persona_criteria,
    cache_file="cached_criteria/persona_eval_criteria_gpt4o.json"
)

print(f"Generated {len(agent_criteria)} AgentEval criteria")
print(f"Generated {sum(len(criteria) for criteria in persona_criteria.values())} PersonaEval criteria across {len(persona_criteria)} personas")
```

### 2. Quantify Criteria

Load cached criteria and use them to evaluate content:

```python
import sys
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Add the package to your path
sys.path.append('../personamatrix-evaluator/src')

# Import the frameworks
from src.core.evaluator import LangGraphAgentEval
from src.core.persona_evaluator import PersonaEvaluator
from src.models.state import Task

load_dotenv()

# Set up your LLM (can be different from criteria generation)
evaluation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Define the same task
task = Task(
    name="Legal Case Summary Utility Evaluation",
    description="""Evaluate utility of legal case summaries for its target user types. 
    Emphasize on whether the summaries is useful to end users, considering the context 
    of the usage, and background of users.""",
    expected_output="Assessment of summary utility and quality"
)

# Load cached criteria
agent_evaluator = LangGraphAgentEval(llm=evaluation_llm)
agent_criteria = agent_evaluator.load_criteria_from_cache(
    task_key="legal_summary_eval_gpt4o",
    cache_file="cached_criteria/agent_eval_criteria_gpt4o.json"
)

persona_evaluator = PersonaEvaluator(llm=evaluation_llm)
persona_criteria = persona_evaluator.load_persona_criteria_from_cache(
    task_key="legal_summary_persona_eval_gpt4o",
    cache_file="cached_criteria/persona_eval_criteria_gpt4o.json"
)

# Your content to evaluate
test_case = """
Smith v. Jones (2023): The plaintiff sued for breach of contract after the defendant 
failed to deliver goods as specified in their agreement. The court ruled in favor 
of the plaintiff, awarding damages of $50,000 plus attorney fees.
"""

# Quantify with AgentEval
agent_result = agent_evaluator.quantify_criteria(
    criteria=agent_criteria,
    task=task,
    test_case=test_case,
    ground_truth="",
    batch_mode=False,
    llm=evaluation_llm
)

# Quantify with PersonaEval
persona_result = persona_evaluator.quantify_persona_criteria(
    persona_criteria=persona_criteria,
    task=task,
    test_case=test_case,
    ground_truth="",
    batch_mode=False,
    llm=evaluation_llm
)

print("AgentEval Results:")
for criterion, scores in agent_result.items():
    print(f"  {criterion}: {scores.get('score', 'N/A')}")

print("\nPersonaEval Results:")
for persona, evaluation in persona_result.items():
    print(f"  {persona}:")
    for criterion, scores in evaluation.items():
        print(f"    {criterion}: {scores.get('score', 'N/A')}")
```

## Usage Examples

### Rate Limiting

Configure rate limiting for API calls to prevent hitting limits:

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

def create_rate_limiter(rpm_limit: int = 15):
    requests_per_second = rpm_limit / 60.0
    return InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=0.5,
        max_bucket_size=rpm_limit
    )

# Use with your LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.8,
    rate_limiter=create_rate_limiter(rpm_limit=30)
)
```

### Batch Processing

For evaluating multiple test cases efficiently:

```python
# Load your test cases
test_cases = [
    "Legal summary 1...",
    "Legal summary 2...",
    "Legal summary 3..."
]

# Evaluate in batch mode
for i, test_case in enumerate(test_cases):
    print(f"Processing case {i+1}/{len(test_cases)}")
    
    agent_result = agent_evaluator.quantify_criteria(
        criteria=agent_criteria,
        task=task,
        test_case=test_case,
        batch_mode=True,  # Enable batch mode
        llm=evaluation_llm
    )
    
    # Process results...
```

## Architecture

The PersonaMatrix Evaluator uses a modular architecture with two main evaluation pipelines:

### AgentEval Pipeline
1. **Task Definition**: Define evaluation task and requirements
2. **Criteria Generation**: Generate generic evaluation criteria using critic agents
3. **Criteria Caching**: Save criteria for reuse across evaluations
4. **Quantification**: Evaluate test cases against the generated criteria

### PersonaEval Pipeline
1. **Persona Loading**: Load predefined personas or define custom ones
2. **Persona-Specific Criteria Generation**: Generate evaluation criteria tailored to each persona
3. **Criteria Caching**: Save persona-specific criteria for reuse
4. **Multi-Persona Quantification**: Evaluate test cases from each persona's perspective

Both pipelines support:
- **LangGraph Workflows**: Orchestrated agent interactions
- **Rate Limiting**: API call management
- **Error Handling**: Robust error recovery and reporting
- **Caching**: Persistent storage of generated criteria

## API Reference

### Core Classes

#### LangGraphAgentEval
The base evaluator for generic task-based evaluation.

**Key Methods:**
- `generate_criteria(task, additional_instructions="", max_round=2, use_subcritic=False)`: Generate evaluation criteria
- `quantify_criteria(criteria, task, test_case, ground_truth="", batch_mode=False, llm=None)`: Evaluate content against criteria
- `save_criteria_to_cache(task_key, criteria, cache_file)`: Save criteria to cache file
- `load_criteria_from_cache(task_key, cache_file)`: Load criteria from cache file

#### PersonaEvaluator
Extends LangGraphAgentEval with persona-specific evaluation capabilities.

**Key Methods:**
- `generate_persona_criteria(task, personas, additional_instructions="", max_round=2, use_subcritic=False)`: Generate persona-specific criteria
- `quantify_persona_criteria(persona_criteria, task, test_case, ground_truth="", batch_mode=False, llm=None)`: Evaluate from multiple persona perspectives
- `save_persona_criteria_to_cache(task_key, persona_criteria, cache_file)`: Save persona criteria to cache
- `load_persona_criteria_from_cache(task_key, cache_file)`: Load persona criteria from cache

### Data Models

#### Task
```python
@dataclass
class Task:
    name: str
    description: str
    expected_output: str
```

#### Criterion
```python
@dataclass
class Criterion:
    name: str
    description: str
    scale: str = "1-5"
    sub_criteria: Optional[List['Criterion']] = None
```

### Available Personas

The framework includes 6 predefined personas for legal document evaluation, each designed around specific user needs and usage patterns. The complete persona definitions and design rationale can be found in `src/models/persona.py`.

Access predefined personas for legal document evaluation:

```python
from src.models.persona import get_all_personas, get_persona_names, get_persona

# Get all available persona names
persona_names = get_persona_names()
# Returns: ['litigation_professional', 'legal_education', 'journalism_media', 
#          'policy_advocacy', 'public_self_help', 'academic_research']

# Get all persona objects
all_personas = get_all_personas()

# Get specific persona
persona = get_persona("litigation_professional")
print(persona.description)  # Civil rights litigators who need precise legal information
print(persona.usage)        # How this persona uses legal summaries
```

**Available Personas:**
1. **Litigation Professional** - Civil rights litigators needing precise legal information for case preparation
2. **Legal Education Community** - Law professors, teachers, and students who need educational content
3. **Journalism & Media** - Court reporters and legal affairs editors who need accessible information
4. **Policy & Advocacy Stakeholders** - Civil rights NGOs and think tanks focused on policy implications
5. **Public Self-Help Users** - Pro se litigants and individuals representing themselves
6. **Academic & Data-Science Researchers** - Empirical legal scholars focused on data analysis

Each persona includes detailed informational needs, evaluation dimensions, and usage patterns. See `src/models/persona.py` for complete specifications.

### Utility Functions

#### Rate Limiter Setup
```python
from langchain_core.rate_limiters import InMemoryRateLimiter

def create_rate_limiter(rpm_limit: int = 15):
    requests_per_second = rpm_limit / 60.0
    return InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=0.5,
        max_bucket_size=rpm_limit
    )
```
