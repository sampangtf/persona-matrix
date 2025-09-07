"""
LangGraph-based AgentEval Framework

A modern implementation of the AgentEval framework using LangGraph for workflow orchestration,
state management, and multi-agent coordination.

Key Features:
- LangGraph workflow orchestration
- Stateful evaluation processes
- Multi-agent coordination (Critic, SubCritic, Quantifier)
- PersonaEval: Multi-persona evaluation framework for legal documents
- Comprehensive error handling and retry logic
- Support for multiple LLM providers (Gemini, Claude, GPT-4)
- Batch evaluation capabilities
- Rich testing and validation suite

Usage:
    from src.core.evaluator import LangGraphAgentEval, quick_evaluate
    from src.core.persona_evaluator import PersonaEvaluator, quick_persona_evaluate
    
    # Quick evaluation
    result = quick_evaluate(
        task_name="Math Problem Solving",
        task_description="Evaluate arithmetic skills",
        test_case="Problem: 2+2. Answer: 4",
        ground_truth="4"
    )
    
    # PersonaEval for legal documents
    persona_result = quick_persona_evaluate(
        task_name="Legal Case Summary Evaluation",
        task_description="Evaluate legal case summary from multiple perspectives",
        test_case="Your legal case summary here",
        persona_names=["litigation_professional", "journalism_media", "public_self_help"]
    )
"""

from .core import LangGraphAgentEval, quick_evaluate
from .core.persona_evaluator import PersonaEvaluator, quick_persona_evaluate
from .agents import CriticAgent, QuantifierAgent, SubCriticAgent, BaseAgent, PersonaCriticAgent, create_persona_critic_from_name
from .models import Task, Criterion
from .models.persona import get_all_personas, get_persona_names, get_persona, LEGAL_DOCUMENT_PERSONAS
from .workflows import create_criteria_generation_workflow, create_quantification_workflow
from .utils import setup_logging, get_default_llm_config, format_evaluation_summary

__version__ = "1.0.0"
__author__ = "LangGraph AgentEval Team"

__all__ = [
    # Core evaluation classes
    "LangGraphAgentEval",
    "quick_evaluate",
    
    # PersonaEval framework
    "PersonaEvaluator",
    "quick_persona_evaluate",
    
    # Agents
    "CriticAgent",
    "QuantifierAgent", 
    "SubCriticAgent",
    "BaseAgent",
    "PersonaCriticAgent",
    "create_persona_critic_from_name",
    
    # Data models
    "EvaluationState",
    "Task",
    "Criterion",
    "EvaluationResult",
    
    # Persona models
    "get_all_personas",
    "get_persona_names", 
    "get_persona",
    "LEGAL_DOCUMENT_PERSONAS",
    
    # Workflows
    "create_criteria_generation_workflow",
    "create_quantification_workflow",
    
    # Utilities
    "setup_logging",
    "get_default_llm_config",
    "format_evaluation_summary"
]