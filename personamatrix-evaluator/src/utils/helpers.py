"""
Utility functions for LangGraph-based AgentEval framework.
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

def setup_logging(level: str = "INFO") -> None:
    """Setup logging for the framework."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def get_default_llm_config() -> Dict[str, Any]:
    """Get default LLM configuration."""
    return {
        "model": "gemini-2.0-flash-thinking-exp-01-21",
        "temperature": 0.7,
        "max_tokens": 4000
    }


def validate_llm_config(config: Dict[str, Any]) -> bool:
    """Validate LLM configuration."""
    required_fields = ["model"]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in LLM config: {field}")
    
    if "temperature" in config:
        temp = config["temperature"]
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
            raise ValueError("Temperature must be a number between 0 and 2")
    
    return True


def format_evaluation_summary(result) -> str:
    """Format evaluation result as a readable summary."""
    summary = f"""
    🎯 Evaluation Summary
    ==================
    
    Task: {result.task.name}
    Workflow ID: {result.workflow_id}
    
    📊 Results:
    - Criteria Generated: {len(result.criteria)}
    - Sub-criteria: {sum(len(c.sub_criteria) for c in result.criteria)}
    - Final Score: {result.final_score}/10 {'' if result.final_score else '(Not available)'}
    - Status: {'✅ Success' if len(result.errors) == 0 else '❌ Has Errors'}
    
    🎯 Criteria Overview:
    """
    
    for i, criterion in enumerate(result.criteria, 1):
        summary += f"\n    {i}. {criterion.name}"
        if criterion.sub_criteria:
            summary += f" ({len(criterion.sub_criteria)} sub-criteria)"
    
    if result.errors:
        summary += f"\n\n❌ Errors ({len(result.errors)}):"
        for error in result.errors:
            summary += f"\n    - {error}"
    
    return summary


def extract_json_from_text(text: str, json_type: str = "object") -> Optional[Dict]:
    """Extract JSON from text response."""
    try:
        if json_type == "array":
            start_char, end_char = "[", "]"
        else:
            start_char, end_char = "{", "}"
        
        start_idx = text.find(start_char)
        end_idx = text.rfind(end_char) + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = text[start_idx:end_idx]
            return json.loads(json_str)
        
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def create_sample_evaluations() -> list:
    """Create sample evaluation cases for testing."""
    return [
        {
            "task_name": "Math Problem Solving",
            "task_description": "Evaluate basic arithmetic skills",
            "test_case": "Problem: 15 + 7 × 3. Answer: 36 (following order of operations)",
            "ground_truth": "36"
        },
        {
            "task_name": "Text Comprehension",
            "task_description": "Evaluate reading comprehension ability",
            "test_case": "Passage: 'The sun is a star.' Question: What is the sun? Answer: The sun is a star.",
            "ground_truth": "Correct identification of the sun as a star"
        },
        {
            "task_name": "Code Review",
            "task_description": "Evaluate code review quality",
            "test_case": "Code: 'def add(a, b): return a + b' Review: 'Function is correct but needs type hints and docstring.'",
            "ground_truth": "Good review should identify missing documentation and type annotations"
        }
    ]