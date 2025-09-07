"""
PersonaEval: Multi-persona evaluation framework.
"""
from typing import Dict, List, Optional, Any, Tuple, Union, Literal
from dataclasses import dataclass
import json
import uuid
import logging
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field
from langchain.output_parsers import RetryOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel

from ..agents.persona_critic_agent import PersonaCriticAgent, create_persona_critic_from_name
from ..agents import QuantifierAgent, SubCriticAgent
from ..models.state import Task, Criterion
from ..models.persona import get_all_personas, get_persona_names, get_persona, Persona
from ..utils.batch_collector import BatchInterrupt
from ..workflows.persona_evaluation_workflow import (
    create_persona_criteria_generation_workflow,
    create_persona_quantification_workflow
)
from .evaluator import LangGraphAgentEval

logger = logging.getLogger(__name__)


@dataclass
class PersonaEvaluationResult:
    """Results from evaluating a single persona."""
    persona_name: str
    persona_description: str
    criteria: List[Criterion]
    quantification_results: Dict[str, Any]
    final_score: Optional[float]
    errors: List[str]
    raw_responses: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "persona_name": self.persona_name,
            "persona_description": self.persona_description,
            "criteria": [criterion.to_dict() for criterion in self.criteria],
            "quantification_results": self.quantification_results,
            "final_score": self.final_score,
            "errors": self.errors,
            "raw_responses": self.raw_responses
        }


@dataclass
class PersonaEvalResult:
    """Complete results from multi-persona evaluation."""
    workflow_id: str
    task: Task
    test_case: str
    ground_truth: Optional[str]
    persona_results: Dict[str, PersonaEvaluationResult]
    summary_report: str
    overall_statistics: Dict[str, Any]
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "workflow_id": self.workflow_id,
            "task": {
                "name": self.task.name,
                "description": self.task.description,
                "expected_output": self.task.expected_output
            },
            "test_case": self.test_case,
            "ground_truth": self.ground_truth,
            "persona_results": {name: result.to_dict() for name, result in self.persona_results.items()},
            "summary_report": self.summary_report,
            "overall_statistics": self.overall_statistics,
            "errors": self.errors
        }
    
    def summary(self) -> str:
        """Generate a summary of the persona evaluation results."""
        summary = f"PersonaEval Results for: {self.task.name}\n"
        summary += f"Workflow ID: {self.workflow_id}\n"
        summary += f"Personas Evaluated: {len(self.persona_results)}\n"
        
        if self.overall_statistics.get("overall_average_score") is not None:
            summary += f"Overall Average Score: {self.overall_statistics['overall_average_score']:.2f}/10\n"
        
        if self.errors:
            summary += f"Total Errors: {len(self.errors)}\n"
        
        return summary


class PersonaEvaluator(LangGraphAgentEval):
    """
    Multi-persona evaluation framework extending LangGraphAgentEval.
    
    Provides persona-specific evaluation capabilities using the PersonaEval approach,
    where criteria generation and quantification are performed from multiple persona perspectives.
    """
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None, llm: Optional[BaseChatModel] = None):
        """
        Initialize the persona evaluator with LLM configuration or instance.
        
        Args:
            llm_config: Configuration for the LLM (model, temperature, etc.)
            llm: Pre-initialized LLM instance to use instead of creating from config
        """
        super().__init__(llm_config=llm_config, llm=llm)
        self.persona_results_history = []  # Store persona-specific evaluation history
        self.quantification_results = []  # Store individual quantification results
        self.persona_cached_criteria = {}  # For caching persona-specific criteria
    
    def generate_persona_criteria(
        self,
        task: Task,
        personas: List[Persona],
        additional_instructions: str = "",
        max_round: int = 2,
        use_subcritic: bool = False,
        llm_config: Optional[Dict[str, Any]] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> Dict[str, List[Criterion]]:
        """
        Generate evaluation criteria for multiple personas.
        
        Args:
            task: The task to evaluate
            personas: List of personas to generate criteria for
            additional_instructions: Additional instructions for the criteria agents
            max_round: Maximum number of rounds for conversation
            use_subcritic: Whether to use the subcritic agent
            llm_config: Optional LLM config to override instance config
            llm: Optional LLM instance to override instance LLM
            
        Returns:
            Dictionary mapping persona names to their generated criteria
        """
        workflow = create_persona_criteria_generation_workflow()
        
        if use_subcritic:
            effective_max_round = max_round * 2
        else:
            effective_max_round = max_round

        # Use provided LLM/config or fall back to instance defaults
        effective_llm = llm or self.llm
        effective_llm_config = llm_config or self.llm_config
        
        # Create initial state
        initial_state = {
            "task": task,
            "personas": personas,
            "llm_config": effective_llm_config,
            "llm": effective_llm,
            "additional_instructions": additional_instructions,
            "max_round": effective_max_round,
            "use_subcritic": use_subcritic,
            "workflow_id": str(uuid.uuid4()),
            "errors": [],
            "thread_id": str(uuid.uuid4())
        }
        
        # Run the workflow
        result = workflow.invoke(initial_state, config={"configurable": {"thread_id": initial_state["thread_id"]}})
        
        if result.get("errors"):
            logger.warning(f"Warnings during persona criteria generation: {result['errors']}")
        
        return result.get("persona_criteria", {})
    
    def quantify_persona_criteria(
        self,
        persona_criteria: Dict[str, List[Criterion]],
        task: Task,
        test_case: str,
        ground_truth: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
        llm: Optional[BaseChatModel] = None,
        batch_mode: bool = False,
        batch_collector: Optional[Any] = None,
        test_case_id: Optional[str] = None,
        dimension: Optional[str] = None,
        level: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Quantify performance using persona-specific criteria.
        
        Args:
            persona_criteria: Dictionary mapping persona names to their criteria
            task: The task being evaluated
            test_case: The test case to evaluate
            ground_truth: The expected/correct result
            llm_config: Optional LLM config to override instance config
            llm: Optional LLM instance to override instance LLM
            batch_mode: If True, collect requests for batch processing instead of immediate execution
            batch_collector: Collector for batch requests
            test_case_id: Test case ID for batch request key generation
            dimension: Dimension for batch request key generation
            level: Level for batch request key generation
            
        Returns:
            Dictionary mapping persona names to their quantification results.
            In batch mode, returns batch collection information instead.
        """
        workflow = create_persona_quantification_workflow()
        
        # For batch mode, we need a workflow without checkpointing to avoid serialization issues
        if batch_mode:
            from ..workflows.persona_evaluation_workflow import create_persona_quantification_workflow_no_checkpointing
            workflow = create_persona_quantification_workflow_no_checkpointing()
        
        # Use provided LLM/config or fall back to instance defaults
        effective_llm = llm or self.llm
        effective_llm_config = llm_config or self.llm_config
        
        # Create initial state
        initial_state = {
            "task": task,
            "persona_criteria": persona_criteria,
            "llm_config": effective_llm_config,
            "llm": effective_llm,
            "test_case": test_case,
            "ground_truth": ground_truth,
            "batch_mode": batch_mode,
            "batch_collector": batch_collector,
            "workflow_id": str(uuid.uuid4()),
            "errors": [],
            "thread_id": str(uuid.uuid4())
        }
        
        # Add batch metadata if in batch mode and values provided
        if batch_mode:
            if test_case_id is not None:
                initial_state["test_case_id"] = test_case_id
            if dimension is not None:
                initial_state["dimension"] = dimension
            if level is not None:
                initial_state["level"] = level
        
        # Run the workflow
        try:
            result = workflow.invoke(initial_state, config={"configurable": {"thread_id": initial_state["thread_id"]}})
            
            if result.get("errors"):
                logger.warning(f"Warnings during persona quantification: {result['errors']}")
            
            quantification_results = result.get("persona_quantification_results", {})
            person_raw_responses = result.get("persona_raw_quantification_responses", {})

            # In batch mode, don't store results since no actual evaluation occurred
            if not batch_mode:
                # Store the results for later summary statistics
                evaluation_record = {
                    "task": task,
                    "test_case": test_case,
                    "ground_truth": ground_truth,
                    "quantification_results": quantification_results,
                    "raw_response": person_raw_responses,
                    "timestamp": datetime.now().isoformat()
                }
                self.quantification_results.append(evaluation_record)
            
            return quantification_results
        
        except BatchInterrupt as e:
            # Re-raise BatchInterrupt to stop workflow and return batch info
            return {
                "batch_mode": True,
                "batch_file": e.batch_file,
                "request_count": e.request_count,
                "ground_truth": ground_truth
            }
        except Exception as e:
            # In batch mode, we may get other exceptions too
            if batch_mode:
                return {
                    "batch_mode": True,
                    "message": str(e),
                    "ground_truth": ground_truth,
                    "error": str(e)
                }
            else:
                raise
    
    def save_persona_criteria_to_cache(self, task_key: str, persona_criteria: Dict[str, List[Criterion]], cache_file: Optional[str] = None):
        """
        Save generated persona criteria to cache and optionally to file.
        
        Args:
            task_key: Unique key for the task
            persona_criteria: Dictionary mapping persona names to their criteria
            cache_file: Optional file path to save criteria as JSON
        """
        # Convert persona criteria to dict format for JSON serialization
        persona_criteria_dict = {}
        for persona_name, criteria_list in persona_criteria.items():
            criteria_dict = []
            for criterion in criteria_list:
                crit_dict = {
                    "name": criterion.name,
                    "description": criterion.description
                }
                if hasattr(criterion, 'accepted_values') and criterion.accepted_values:
                    crit_dict["accepted_values"] = criterion.accepted_values
                if hasattr(criterion, 'sub_criteria') and criterion.sub_criteria:
                    crit_dict["sub_criteria"] = []
                    for sub_crit in criterion.sub_criteria:
                        sub_dict = {
                            "name": sub_crit.name,
                            "description": sub_crit.description
                        }
                        if hasattr(sub_crit, 'accepted_values') and sub_crit.accepted_values:
                            sub_dict["accepted_values"] = sub_crit.accepted_values
                        crit_dict["sub_criteria"].append(sub_dict)
                criteria_dict.append(crit_dict)
            persona_criteria_dict[persona_name] = criteria_dict
        
        # Store in memory cache
        self.persona_cached_criteria[task_key] = persona_criteria_dict
        
        # Optionally save to file
        if cache_file:
            cache_data = {
                "task_key": task_key,
                "persona_criteria": persona_criteria_dict,
                "timestamp": datetime.now().isoformat()
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
    
    def load_persona_criteria_from_cache(self, task_key: str, cache_file: Optional[str] = None) -> Optional[Dict[str, List[Criterion]]]:
        """
        Load persona criteria from cache or file.
        
        Args:
            task_key: Unique key for the task
            cache_file: Optional file path to load criteria from JSON
            
        Returns:
            Dictionary mapping persona names to their criteria if found, None otherwise
        """
        persona_criteria_dict = None
        
        # Try to load from file first if provided
        if cache_file and Path(cache_file).exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    if cache_data.get("task_key") == task_key:
                        persona_criteria_dict = cache_data.get("persona_criteria")
                        # Also store in memory cache
                        self.persona_cached_criteria[task_key] = persona_criteria_dict
            except Exception as e:
                logger.warning(f"Failed to load persona criteria from file {cache_file}: {e}")
        
        # Fall back to memory cache
        if persona_criteria_dict is None:
            persona_criteria_dict = self.persona_cached_criteria.get(task_key)
        
        if persona_criteria_dict is None:
            return None
        
        # Convert dict format back to Criterion objects
        persona_criteria = {}
        for persona_name, criteria_dict_list in persona_criteria_dict.items():
            criteria = []
            for crit_dict in criteria_dict_list:
                # Create sub-criteria first if they exist
                sub_criteria = []
                if "sub_criteria" in crit_dict:
                    for sub_dict in crit_dict["sub_criteria"]:
                        sub_criterion = Criterion(
                            name=sub_dict["name"],
                            description=sub_dict["description"],
                            accepted_values=sub_dict.get("accepted_values")
                        )
                        sub_criteria.append(sub_criterion)
                
                criterion = Criterion(
                    name=crit_dict["name"],
                    description=crit_dict["description"],
                    accepted_values=crit_dict.get("accepted_values"),
                    sub_criteria=sub_criteria if sub_criteria else None
                )
                criteria.append(criterion)
            persona_criteria[persona_name] = criteria
        
        return persona_criteria
    
    def _calculate_persona_final_score(self, quantification: Dict[str, Any]) -> Optional[float]:
        """Calculate final score for a persona from quantification results."""
        try:
            evaluations = quantification.get("evaluations", {})
            scores = []
            
            for criterion_name, evaluation_result in evaluations.items():
                if isinstance(evaluation_result, dict):
                    if "numerical_score" in evaluation_result:
                        # Flat criterion
                        scores.append(float(evaluation_result["numerical_score"]))
                    else:
                        # Criterion with sub-criteria
                        sub_scores = []
                        for sub_name, sub_result in evaluation_result.items():
                            if isinstance(sub_result, dict) and "numerical_score" in sub_result:
                                sub_scores.append(float(sub_result["numerical_score"]))
                        
                        if sub_scores:
                            scores.append(sum(sub_scores) / len(sub_scores))
            
            return sum(scores) / len(scores) if scores else None
            
        except Exception as e:
            logger.error(f"Error calculating persona final score: {e}")
            return None
    
    def _calculate_overall_statistics(self, persona_results: Dict[str, PersonaEvaluationResult]) -> Dict[str, Any]:
        """Calculate overall statistics across all personas."""
        scores = []
        persona_scores = {}
        
        for persona_name, result in persona_results.items():
            if result.final_score is not None:
                scores.append(result.final_score)
                persona_scores[persona_name] = result.final_score
        
        if not scores:
            return {"message": "No valid scores available"}
        
        scores_array = np.array(scores)
        
        return {
            "overall_average_score": float(np.mean(scores_array)),
            "score_statistics": {
                "mean": float(np.mean(scores_array)),
                "median": float(np.median(scores_array)),
                "std_dev": float(np.std(scores_array, ddof=1)) if len(scores) > 1 else 0.0,
                "min_score": float(np.min(scores_array)),
                "max_score": float(np.max(scores_array)),
                "score_range": float(np.ptp(scores_array))
            },
            "persona_scores": persona_scores,
            "num_personas": len(persona_results),
            "personas_with_scores": len(scores)
        }
    
    def _generate_summary_report(self, persona_results: Dict[str, PersonaEvaluationResult], overall_stats: Dict[str, Any]) -> str:
        """Generate a comprehensive summary report."""
        report = "PersonaEval Summary Report\n"
        report += "=" * 50 + "\n\n"
        
        # Overall statistics
        if "overall_average_score" in overall_stats:
            report += f"Overall Average Score: {overall_stats['overall_average_score']:.2f}/10\n"
            report += f"Score Range: {overall_stats['score_statistics']['min_score']:.2f} - {overall_stats['score_statistics']['max_score']:.2f}\n"
            report += f"Standard Deviation: {overall_stats['score_statistics']['std_dev']:.2f}\n\n"
        
        # Individual persona results
        report += "Individual Persona Results:\n"
        report += "-" * 30 + "\n"
        
        for persona_name, result in persona_results.items():
            report += f"\n{persona_name}:\n"
            if result.final_score is not None:
                report += f"  Score: {result.final_score:.2f}/10\n"
            else:
                report += f"  Score: N/A\n"
            report += f"  Criteria Count: {len(result.criteria)}\n"
            if result.errors:
                report += f"  Errors: {len(result.errors)}\n"
        
        return report
    
    def _add_persona_evaluation_result(self, result: PersonaEvalResult) -> None:
        """Add a persona evaluation result to the history."""
        evaluation_record = {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        self.persona_results_history.append(evaluation_record)
    
    def get_persona_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics for all persona evaluations."""
        if not self.quantification_results:
            return {"message": "No persona evaluations completed yet"}
        
        total_evaluations = len(self.quantification_results)
        all_persona_scores = defaultdict(list)  # {persona_name: [scores]}
        all_test_case_scores = []
        
        # Aggregate scores across all evaluations
        for evaluation in self.quantification_results:
            quantification_results = evaluation["quantification_results"]
            test_case_scores = []
            
            for persona_name, quantification_result in quantification_results.items():
                if 'error' not in quantification_result:
                    # Calculate final score for this persona
                    final_score = self._calculate_persona_final_score(quantification_result)
                    if final_score is not None:
                        all_persona_scores[persona_name].append(final_score)
                        test_case_scores.append(final_score)
            
            # Overall test case score (average across personas for this evaluation)
            if test_case_scores:
                all_test_case_scores.append(sum(test_case_scores) / len(test_case_scores))
        
        # Calculate persona averages
        persona_averages = {}
        for persona_name, scores in all_persona_scores.items():
            if scores:
                scores_array = np.array(scores)
                persona_averages[persona_name] = {
                    "average_score": float(np.mean(scores_array)),
                    "num_evaluations": len(scores),
                    "min_score": float(np.min(scores_array)),
                    "max_score": float(np.max(scores_array)),
                    "std_dev": float(np.std(scores_array, ddof=1)) if len(scores) > 1 else 0.0,
                    "all_scores": scores
                }
        
        # Overall statistics
        summary = {
            "total_persona_evaluations": total_evaluations,
            "persona_averages": persona_averages,
            "llm_config": self.llm_config,
            "timestamp": datetime.now().isoformat()
        }
        
        if all_test_case_scores:
            test_case_scores_array = np.array(all_test_case_scores)
            summary["overall_test_case_statistics"] = {
                "mean": float(np.mean(test_case_scores_array)),
                "median": float(np.median(test_case_scores_array)),
                "std_dev": float(np.std(test_case_scores_array, ddof=1)) if len(all_test_case_scores) > 1 else 0.0,
                "min_score": float(np.min(test_case_scores_array)),
                "max_score": float(np.max(test_case_scores_array)),
                "all_scores": all_test_case_scores
            }
        
        return summary
    
    def clear_persona_history(self) -> None:
        """Clear the persona evaluation history."""
        self.persona_results_history = []
        self.quantification_results = []


def quick_persona_evaluate(
    task_name: str,
    task_description: str,
    test_case: str,
    persona_names: List[str],
    llm_config: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[str] = None,
    use_subcritic: bool = False
) -> PersonaEvalResult:
    """
    Quick persona evaluation function for simple use cases.
    
    Args:
        task_name: Name of the task
        task_description: Description of the task
        test_case: Test case to evaluate
        persona_names: List of persona names to use
        llm_config: LLM configuration (uses default if None)
        ground_truth: Optional ground truth
        use_subcritic: Whether to use subcritic refinement
        
    Returns:
        PersonaEvalResult with multi-persona evaluation results
    """
    if llm_config is None:
        llm_config = {
            "model": "gemini-2.0-flash",
            "temperature": 0.7
        }
    
    # Create task
    task = Task(
        name=task_name,
        description=task_description,
        expected_output="Persona evaluation completed"
    )
    
    # Get personas
    personas = []
    for persona_name in persona_names:
        persona = get_persona(persona_name)
        if persona:
            personas.append(persona)
        else:
            logger.warning(f"Persona '{persona_name}' not found, skipping")
    
    if not personas:
        raise ValueError("No valid personas found")
    
    # Initialize evaluator and run evaluation
    evaluator = PersonaEvaluator(llm_config=llm_config)
    
    # Generate criteria
    persona_criteria = evaluator.generate_persona_criteria(
        task=task,
        personas=personas,
        use_subcritic=use_subcritic
    )
    
    # Quantify criteria 
    quantification_results = evaluator.quantify_persona_criteria(
        persona_criteria=persona_criteria,
        task=task,
        test_case=test_case,
        ground_truth=ground_truth or ""
    )
    
    # Create persona evaluation results
    persona_results = {}
    for persona_name, quantification in quantification_results.items():
        final_score = evaluator._calculate_persona_final_score(quantification)
        persona_results[persona_name] = PersonaEvaluationResult(
            persona_name=persona_name,
            persona_description=next(p.description for p in personas if p.name == persona_name),
            criteria=persona_criteria.get(persona_name, []),
            quantification_results=quantification,
            final_score=final_score,
            errors=[],
            raw_responses={}
        )
    
    # Calculate overall statistics
    overall_stats = evaluator._calculate_overall_statistics(persona_results)
    summary_report = evaluator._generate_summary_report(persona_results, overall_stats)
    
    # Create final result
    return PersonaEvalResult(
        workflow_id=str(uuid.uuid4()),
        task=task,
        test_case=test_case,
        ground_truth=ground_truth,
        persona_results=persona_results,
        summary_report=summary_report,
        overall_statistics=overall_stats,
        errors=[]
    )
