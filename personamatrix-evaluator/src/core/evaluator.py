"""
Main evaluation runner using LangGraph workflows.
Supports batch request collection for Google Gemini Batch API.
"""
from typing import Dict, Any, Optional, List, Literal, Union
import asyncio
import json
import uuid
import numpy as np
from pathlib import Path
import logging 
from datetime import datetime

from collections import defaultdict
from langchain_core.language_models import BaseChatModel

from ..workflows.evaluation_workflow import (
    create_criteria_generation_workflow, 
    create_quantification_workflow
)
from ..models.state import Task, Criterion
from ..agents import CriticAgent, SubCriticAgent, QuantifierAgent
from ..utils.batch_collector import BatchRequestCollector, BatchInterrupt
from datetime import datetime
 
logger = logging.getLogger(__name__)

def generate_criteria(
    llm_config: Optional[Union[Dict, Literal[False]]] = None,
    task: Task = None,
    additional_instructions: str = "",
    max_round: int = 2,
    use_subcritic: bool = False,
    llm: Optional[BaseChatModel] = None,
) -> List[Criterion]:
    """
    Creates a list of criteria for evaluating the utility of a given task.
    
    Args:
        llm_config (dict or bool): llm inference configuration.
        task (Task): The task to evaluate.
        additional_instructions (str): Additional instructions for the criteria agent.
        max_round (int): The maximum number of rounds to run the conversation.
        use_subcritic (bool): Whether to use the subcritic agent to generate subcriteria.
        llm (BaseChatModel, optional): Pre-initialized LLM instance to use instead of creating from config.
        
    Returns:
        List[Criterion]: A list of Criterion objects for evaluating the utility of the given task.
    """
    workflow = create_criteria_generation_workflow()
    
    if use_subcritic:
        effective_max_round = max_round * 2
    else:
        effective_max_round = max_round

    # Create initial state
    initial_state = {
        "task": task,
        "llm_config": llm_config,
        "llm": llm,
        "additional_instructions": additional_instructions,
        "max_round": effective_max_round,
        "use_subcritic": use_subcritic,
        "workflow_id": str(uuid.uuid4()),
        "errors": [],
        "round_count": 0,
        "thread_id": str(uuid.uuid4())  # Unique identifier for the session
    }
    
    # Run the workflow
    result = workflow.invoke(initial_state, config={"configurable": {"thread_id": initial_state["thread_id"]}})
    
    if result.get("errors"):
        # Log errors but still return criteria if available
        print(f"Warnings during criteria generation: {result['errors']}")
    
    return result.get("criteria", [])


def quantify_criteria(
    llm_config: Optional[Union[Dict, Literal[False]]] = None,
    criteria: List[Criterion] = None,
    task: Task = None,
    test_case: str = "",
    ground_truth: str = "",
    llm: Optional[BaseChatModel] = None,
    batch_mode: bool = False,
    batch_collector: Optional[Any] = None,
    test_case_id: Optional[str] = None,
    dimension: Optional[str] = None,
    level: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Quantifies the performance of a system using the provided criteria.
    
    Args:
        llm_config (dict or bool): llm inference configuration.
        criteria (List[Criterion]): A list of criteria for evaluating the utility of a given task.
        task (Task): The task to evaluate.
        test_case (str): The test case to evaluate.
        ground_truth (str): The ground truth for the test case.
        llm (BaseChatModel, optional): Pre-initialized LLM instance to use instead of creating from config.
        batch_mode (bool): If True, collect requests for batch processing instead of immediate execution.
        batch_collector (BatchRequestCollector, optional): Collector for batch requests.
        test_case_id (str, optional): Test case ID for batch request key generation.
        dimension (str, optional): Dimension for batch request key generation.
        level (int, optional): Level for batch request key generation.
        
    Returns:
        dict: A dictionary where the keys are the criteria and the values are the assessed 
              performance based on accepted values for each criteria. In batch mode, returns
              batch collection information instead.
    """
    workflow = create_quantification_workflow()
    
    # For batch mode, we need a workflow without checkpointing to avoid serialization issues
    if batch_mode:
        # Create workflow without checkpointer for batch mode
        from ..workflows.evaluation_workflow import create_quantification_workflow_no_checkpointing
        workflow = create_quantification_workflow_no_checkpointing()
    
    # Create initial state
    initial_state = {
        "task": task,
        "criteria": criteria,
        "llm_config": llm_config,
        "llm": llm,  # Pass the LLM instance
        "test_case": test_case,
        "ground_truth": ground_truth,
        "batch_mode": batch_mode,
        "batch_collector": batch_collector,
        "workflow_id": str(uuid.uuid4()),
        "errors": [],
        "thread_id": str(uuid.uuid4()),  # Unique identifier for the session
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
            # Log errors but still return results if available
            print(f"Warnings during quantification: {result['errors']}")
        
        # In batch mode, return the quantification results directly (which contains batch info)
        if batch_mode:
            return result.get("quantification_results", {})
        else:
            return {
                "actual_success": ground_truth, 
                "estimated_performance": result.get("quantification_results", {}),
                "raw_quantification_response": result.get("raw_quantification_response", "")
            }
    
    except BatchInterrupt as e:
        # Re-raise BatchInterrupt to stop workflow and return batch info
        return {
            "batch_mode": True,
            "batch_file": e.batch_file,
            "request_count": e.request_count,
            "actual_success": ground_truth
        }
    except Exception as e:
        # In batch mode, we may get other exceptions too
        if batch_mode:
            # Return batch collection status without raising
            return {
                "batch_mode": True,
                "message": str(e),
                "actual_success": ground_truth,
                "error": str(e)
            }
        else:
            raise


class LangGraphAgentEval:
    """
    Main class for running agent evaluations using LangGraph workflows.
    Provides convenience methods for the decoupled evaluation functions.
    """
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None, llm: Optional[BaseChatModel] = None):
        """
        Initialize the evaluator with LLM configuration or instance.
        
        Args:
            llm_config: Configuration for the LLM (model, temperature, etc.)
            llm: Pre-initialized LLM instance to use instead of creating from config
        """
        if llm is not None:
            self.llm = llm
            self.llm_config = llm_config  # Store config for batch operations even when using pre-initialized LLM
        elif llm_config is not None:
            self.llm_config = llm_config
            self.llm = self._create_llm_from_config(llm_config)
        else:
            raise ValueError("Either llm_config or llm instance must be provided")
        
        self.results_history = []
        self.cached_criteria = {}  # For caching criteria by task
    
    def _create_llm_from_config(self, llm_config: Dict[str, Any]) -> BaseChatModel:
        """Create LangChain LLM from configuration - reuse BaseAgent logic."""
        from ..agents.base_agent import BaseAgent
        # Create a temporary BaseAgent instance just to use its LLM creation logic
        temp_agent = BaseAgent("temp", "temp", llm_config)
        return temp_agent.llm
    
    def save_criteria_to_cache(self, task_key: str, criteria: List[Criterion], cache_file: Optional[str] = None):
        """
        Save generated criteria to cache and optionally to file.
        
        Args:
            task_key: Unique key for the task
            criteria: List of criteria to cache
            cache_file: Optional file path to save criteria as JSON
        """
        # Convert criteria to dict format for JSON serialization
        criteria_dict = []
        for criterion in criteria:
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
        
        # Store in memory cache
        self.cached_criteria[task_key] = criteria_dict
        
        # Optionally save to file
        if cache_file:
            cache_data = {
                "task_key": task_key,
                "criteria": criteria_dict,
                "timestamp": datetime.now().isoformat()
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
    
    def load_criteria_from_cache(self, task_key: str, cache_file: Optional[str] = None) -> Optional[List[Criterion]]:
        """
        Load criteria from cache or file.
        
        Args:
            task_key: Unique key for the task
            cache_file: Optional file path to load criteria from JSON
            
        Returns:
            List of criteria if found, None otherwise
        """
        criteria_dict = None
        
        # Try to load from file first if provided
        if cache_file and Path(cache_file).exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    if cache_data.get("task_key") == task_key:
                        criteria_dict = cache_data.get("criteria")
                        # Also store in memory cache
                        self.cached_criteria[task_key] = criteria_dict
            except Exception as e:
                logger.warning(f"Failed to load criteria from file {cache_file}: {e}")
        
        # Fall back to memory cache
        if criteria_dict is None:
            criteria_dict = self.cached_criteria.get(task_key)
        
        if criteria_dict is None:
            return None
        
        # Convert dict format back to Criterion objects
        criteria = []
        for crit_dict in criteria_dict:
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
        
        return criteria
    
    def generate_criteria(
        self,
        task: Task,
        additional_instructions: str = "",
        max_round: int = 2,
        use_subcritic: bool = False,
        llm_config: Optional[Dict[str, Any]] = None,
        llm: Optional[BaseChatModel] = None,
    ) -> List[Criterion]:
        """
        Generate evaluation criteria for a task.
        
        Args:
            task: The task to evaluate
            additional_instructions: Additional instructions for the criteria agent
            max_round: Maximum number of rounds for conversation
            use_subcritic: Whether to use the subcritic agent
            llm_config: Optional LLM config to override instance config
            llm: Optional LLM instance to override instance LLM
            
        Returns:
            List of generated criteria

        Example outputs:
            [
                Criterion(
                    name="criterion_name", 
                    description="Criterion description"
                    accepted_values=[...]  # Optional, if applicable
                    sub_criteria=[Criterion(...) if applicable]
                )
                Criterion(...), 
                ...
            ]
        """
        # Use provided LLM/config or fall back to instance defaults
        effective_llm = llm or self.llm
        effective_llm_config = llm_config or self.llm_config
        
        return generate_criteria(
            llm_config=effective_llm_config,
            task=task,
            additional_instructions=additional_instructions,
            max_round=max_round,
            use_subcritic=use_subcritic,
            llm=effective_llm
        )
    
    def quantify_criteria(
        self,
        criteria: List[Criterion],
        task: Task,
        test_case: str,
        ground_truth: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
        llm: Optional[BaseChatModel] = None,
        batch_mode: bool = False,
        batch_collector: Optional[BatchRequestCollector] = None,
        test_case_id: Optional[str] = None,
        dimension: Optional[str] = None,
        level: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Quantify performance using provided criteria.
        
        Args:
            criteria: List of criteria to evaluate against
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
            Dictionary with quantification results. In batch mode, returns batch collection information.
        
        Example outputs:
            {
                "actual_success": ground_truth,
                "estimated_performance": {
                    "evaluations": {
                        "criterion_name": {
                            "numerical_score": float,
                            "performance": "assessment"
                        },
                        ...
                    }
                }
            }   
        """
        # Use provided LLM/config or fall back to instance defaults
        effective_llm = llm or self.llm
        effective_llm_config = llm_config or self.llm_config
        
        result = quantify_criteria(
            llm_config=effective_llm_config,
            criteria=criteria,
            task=task,
            test_case=test_case,
            ground_truth=ground_truth,
            llm=effective_llm,
            batch_mode=batch_mode,
            batch_collector=batch_collector,
            test_case_id=test_case_id,
            dimension=dimension,
            level=level
        )
        
        # In batch mode, don't store in history since no actual evaluation occurred
        if not batch_mode:
            # Store result in history for summary statistics
            evaluation_record = {
                "task": task,
                "criteria": criteria,
                "test_case": test_case,
                "ground_truth": ground_truth,
                "result": result,
                "timestamp": datetime.now().isoformat() # Store as ISO 8601 string
            }
            self.results_history.append(evaluation_record)
        
        return result
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics for all evaluations.
        
        This method extracts scores for each criteria and subcriteria, computes criteria 
        average numerical scores, then calculates test case average scores and exports 
        comprehensive summary statistics.
        """
        if not self.results_history:
            return {"message": "No evaluations completed yet"}
        
        total_evaluations = len(self.results_history)
        all_test_case_scores = []
        all_test_case_details = []  # Store detailed info for each test case
        
        # Aggregators for criteria and subcriteria scores
        criteria_score_aggregator = defaultdict(list)  # {criteria_name: [scores]}
        subcriteria_score_aggregator = defaultdict(lambda: defaultdict(list)) # {criteria_name: {subcriteria_name: [scores]}}
        
        # Iterate through all test cases to extract scores
        for eval_idx, evaluation in enumerate(self.results_history):
            estimated_performance = evaluation["result"].get("estimated_performance", {}).get("evaluations", {})
            
            # Skip evaluations without proper quantification results
            if not estimated_performance:
                logger.warning(f"Skipping evaluation {eval_idx} due to missing or invalid quantification results")
                continue
        
            test_case_scores = []  # Scores of all criteria for this specific test case
            test_case_criteria_breakdown = {}  # Track individual criteria scores for this test case
            
            for criteria_name, evaluation_result in estimated_performance.items():
                criteria_final_score = None
                
                # Handle flat criteria (direct numerical_score)
                if isinstance(evaluation_result, dict) and "numerical_score" in evaluation_result:
                    criteria_final_score = float(evaluation_result["numerical_score"])
                    test_case_scores.append(criteria_final_score)
                    test_case_criteria_breakdown[criteria_name] = criteria_final_score
                    
                    # Aggregate by criteria name
                    criteria_score_aggregator[criteria_name].append(criteria_final_score)
                
                # Handle criteria with sub-criteria
                elif isinstance(evaluation_result, dict):
                    criteria_subscores = []
                    subcriteria_breakdown = {} 
                    
                    # Extract all subcriteria scores
                    for subcriteria_name, subcriteria_result in evaluation_result.items():
                        if isinstance(subcriteria_result, dict) and "numerical_score" in subcriteria_result:
                            subscore = float(subcriteria_result["numerical_score"])
                            criteria_subscores.append(subscore)
                            subcriteria_breakdown[subcriteria_name] = subscore
                            
                            # Aggregate by subcriteria name 
                            subcriteria_score_aggregator[criteria_name][subcriteria_name].append(subscore)
                    
                    # Calculate average for this criteria across its subcriteria
                    if criteria_subscores:
                        criteria_final_score = sum(criteria_subscores) / len(criteria_subscores)
                        test_case_scores.append(criteria_final_score)
                        test_case_criteria_breakdown[criteria_name] = {
                            "average_score": criteria_final_score,
                            "subcriteria_scores": subcriteria_breakdown
                        }
                        
                        # Aggregate criteria average
                        criteria_score_aggregator[criteria_name].append(criteria_final_score)
            
            # Calculate test case average score
            if test_case_scores:
                # Calculate average score for this test case
                test_case_avg = sum(test_case_scores) / len(test_case_scores)
                all_test_case_scores.append(test_case_avg)
                
                # Store detailed test case information
                test_case_detail = {
                    "test_case_index": eval_idx,
                    "test_case_description": evaluation.get("test_case", "Unknown")[:100] + "..." if len(evaluation.get("test_case", "")) > 100 else evaluation.get("test_case", "Unknown"),
                    "test_case_average_score": test_case_avg,
                    "num_criteria": len(test_case_scores),
                    "criteria_breakdown": test_case_criteria_breakdown,
                    "individual_criteria_scores": test_case_scores,
                    "timestamp": evaluation.get("timestamp", "Unknown")
                }
                all_test_case_details.append(test_case_detail)
        
        # Compute comprehensive summary statistics
        summary = {
            "total_evaluations": total_evaluations,
            "evaluations_with_scores": len(all_test_case_scores),
            "llm_config": self.llm_config,
            "timestamp": str(uuid.uuid4()),  # Summary generation timestamp
        }
        
        # Overall test case averages and statistics
        if all_test_case_scores:
            test_case_scores_array = np.array(all_test_case_scores)
            summary["overall_test_case_average"] = float(np.mean(test_case_scores_array))
            summary["test_case_scores"] = all_test_case_scores
            summary["test_case_score_statistics"] = {
            "mean": float(np.mean(test_case_scores_array)),
            "median": float(np.median(test_case_scores_array)),
            "std_dev": float(np.std(test_case_scores_array, ddof=1)) if len(all_test_case_scores) > 1 else 0.0,
            "min_score": float(np.min(test_case_scores_array)),
            "max_score": float(np.max(test_case_scores_array)),
            "score_range": float(np.ptp(test_case_scores_array))
            }
            summary["detailed_test_cases"] = all_test_case_details
        
        # Comprehensive criteria averages and statistics
        criteria_averages = {}
        for criteria_name, scores in criteria_score_aggregator.items():
            if scores:
                scores_array = np.array(scores)
                criteria_averages[criteria_name] = {
                    "average_score": float(np.mean(scores_array)),
                    "num_evaluations": len(scores),
                    "min_score": float(np.min(scores_array)),
                    "max_score": float(np.max(scores_array)),
                    "score_range": float(np.ptp(scores_array)),
                    "std_dev": float(np.std(scores_array, ddof=1)) if len(scores) > 1 else 0.0,
                    "median_score": float(np.median(scores_array)),
                    "all_scores": scores
                }
        summary["criteria_averages"] = criteria_averages
        
        # Comprehensive subcriteria averages and statistics
        subcriteria_averages = {}
        for criteria_name, subcriteria_dict in subcriteria_score_aggregator.items():
            if subcriteria_dict:  # Only include criteria that have subcriteria
                subcriteria_averages[criteria_name] = {}
                for subcriteria_name, scores in subcriteria_dict.items():
                    if scores:
                        scores_array = np.array(scores)
                        subcriteria_averages[criteria_name][subcriteria_name] = {
                            "average_score": float(np.mean(scores_array)),
                            "num_evaluations": len(scores),
                            "min_score": float(np.min(scores_array)),
                            "max_score": float(np.max(scores_array)),
                            "score_range": float(np.ptp(scores_array)),
                            "std_dev": float(np.std(scores_array, ddof=1)) if len(scores) > 1 else 0.0,
                            "median_score": float(np.median(scores_array)),
                            "all_scores": scores
                        }
        summary["subcriteria_averages"] = subcriteria_averages
        
        # Summary performance insights
        summary["performance_insights"] = {
            "highest_performing_criteria": max(criteria_averages.items(), key=lambda x: x[1]["average_score"])[0] if criteria_averages else None,
            "lowest_performing_criteria": min(criteria_averages.items(), key=lambda x: x[1]["average_score"])[0] if criteria_averages else None,
            "most_consistent_criteria": min(criteria_averages.items(), key=lambda x: x[1]["std_dev"])[0] if criteria_averages and len(all_test_case_scores) > 1 else None,
            "most_variable_criteria": max(criteria_averages.items(), key=lambda x: x[1]["std_dev"])[0] if criteria_averages and len(all_test_case_scores) > 1 else None,
            "best_test_case_index": int(np.argmax(test_case_scores_array)) if all_test_case_scores else None,
            "worst_test_case_index": int(np.argmin(test_case_scores_array)) if all_test_case_scores else None
        }
        
        return summary
    
    def add_evaluation_result(
        self,
        task: Task,
        criteria: List[Criterion],
        test_case: str,
        result: Dict[str, Any],
        ground_truth: str = ""
    ) -> None:
        """
        Add an evaluation result to the history.
        
        Args:
            task: The task that was evaluated
            criteria: The criteria used for evaluation
            test_case: The test case that was evaluated
            result: The evaluation result
            ground_truth: The ground truth for the test case
        """
        evaluation_record = {
            "task": task,
            "criteria": criteria,
            "test_case": test_case,
            "ground_truth": ground_truth,
            "result": result,
            "timestamp": str(uuid.uuid4())
        }
        self.results_history.append(evaluation_record)
    
    def clear_history(self) -> None:
        """Clear the evaluation history."""
        self.results_history = []
    
    def create_batch_collector(self, test_case_id: str = None, provider: str = "gemini") -> BatchRequestCollector:
        """Create a batch collector for collecting evaluation requests."""
        return BatchRequestCollector(
            provider=provider, 
            test_case_id=test_case_id,
            llm_config=self.llm_config
        )
    
    def save_batch_requests(self, batch_collector: BatchRequestCollector, filename: str = None) -> str:
        """Save collected batch requests to file."""
        if batch_collector.get_request_count() == 0:
            raise ValueError("No requests collected in batch collector")
        
        batch_file = batch_collector.save_batch_file(filename)
        logger.info(f"Saved {batch_collector.get_request_count()} batch requests to: {batch_file}")
        return batch_file
    
    def collect_multiple_evaluations(
        self,
        evaluations: List[Dict[str, Any]],
        batch_collector: Optional[BatchRequestCollector] = None,
        test_case_id: str = None
    ) -> Dict[str, Any]:
        """
        Collect multiple evaluation requests for batch processing.
        
        Args:
            evaluations: List of evaluation configs, each containing:
                - criteria: List[Criterion]
                - task: Task 
                - test_case: str
                - ground_truth: str (optional)
                - dimension: str (optional)
                - level: int (optional)
            batch_collector: Optional batch collector to use
            test_case_id: Test case ID for batch organization
            
        Returns:
            Dictionary with batch collection results
        """
        if batch_collector is None:
            batch_collector = self.create_batch_collector(test_case_id)
        
        results = {
            "batch_collector": batch_collector,
            "collected_requests": [],
            "errors": []
        }
        
        for i, evaluation in enumerate(evaluations):
            try:
                # Set up state for this evaluation
                state_updates = {
                    "test_case_id": test_case_id,
                    "dimension": evaluation.get("dimension"),
                    "level": evaluation.get("level")
                }
                
                # Call quantify_criteria with batch_mode=True
                result = self.quantify_criteria(
                    criteria=evaluation["criteria"],
                    task=evaluation["task"],
                    test_case=evaluation["test_case"],
                    ground_truth=evaluation.get("ground_truth", ""),
                    batch_mode=True,
                    batch_collector=batch_collector
                )
                
                results["collected_requests"].append({
                    "index": i,
                    "result": result
                })
                
            except Exception as e:
                results["errors"].append({
                    "index": i,
                    "error": str(e)
                })
                logger.error(f"Error collecting evaluation {i}: {e}")
        
        return results


def quick_evaluate(
    task_name: str,
    task_description: str,
    test_case: str,
    llm_config: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[str] = None,
    use_subcritic: bool = False
) -> Dict[str, Any]:
    """
    Quick evaluation function for simple use cases.
    
    Args:
        task_name: Name of the task
        task_description: Description of the task
        test_case: Test case to evaluate
        llm_config: LLM configuration (uses default if None)
        ground_truth: Optional ground truth
        use_subcritic: Whether to use subcritic refinement
        
    Returns:
        Dictionary with criteria and quantification results
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
        expected_output="Evaluation completed"
    )
    
    # Generate criteria
    criteria = generate_criteria(
        llm_config=llm_config,
        task=task,
        use_subcritic=use_subcritic
    )
    
    # Quantify performance
    quantification = quantify_criteria(
        llm_config=llm_config,
        criteria=criteria,
        task=task,
        test_case=test_case,
        ground_truth=ground_truth or ""
    )
    
    return {
        "task": task,
        "criteria": criteria,
        "quantification": quantification
    }