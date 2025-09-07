"""
Data models and state definitions for LangGraph-based AgentEval framework.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import dataclass
import json
import logging

# Configure logging 
logger = logging.getLogger(__name__) 
logger = logging.getLogger(__name__)

# class EvaluationState(TypedDict, total=False):
#     """State object for the evaluation workflow."""
#     # Workflow metadata
#     workflow_id: str
#     current_step: str
#     evaluation_complete: bool
#     errors: List[str]
#     retry_count: int
    
#     # Task definition
#     task_name: str
#     task_description: str
#     test_case: str
#     successful_response: str
#     failed_response: str
#     ground_truth: Optional[str]
    
#     # LLM configuration
#     llm_config: Dict[str, Any]
    
#     # Evaluation criteria
#     criteria: List[Dict[str, Any]]
#     raw_criteria_response: str
    
#     # Quantification results
#     quantification_results: Dict[str, Any]
#     raw_quantification_response: str
#     final_score: Optional[float]
    
#     # Detailed results
#     detailed_results: Dict[str, Any]


@dataclass
class Task:
    """Task definition for evaluation."""
    name: str
    description: str
    expected_output: str
    successful_response: Optional[str] = None
    failed_response: Optional[str] = None
    
    def get_sys_message(self) -> str:
        """Generate system message for the task."""
        message = f"Task: {self.name}\n"
        message += f"Task description: {self.description}\n"
        message += f"Expected output: {self.expected_output}\n"
        
        if self.successful_response:
            message += f"Task successful example: {self.successful_response}\n"
        
        if self.failed_response:
            message += f"Task failed example: {self.failed_response}\n"
        
        return message


@dataclass
class Criterion:
    """Individual evaluation criterion."""
    name: str
    description: str
    accepted_values: Optional[List[str]] = None  # None when sub_criteria exist 
    sub_criteria: Optional[List[Criterion]] = None  # None when no accepted values exist
    
    def __post_init__(self):
        # Enforce mutually exclusive relationship
        if self.accepted_values is not None and (self.sub_criteria is not None or len(self.sub_criteria or []) > 0):
            logger.warning("Cannot have both accepted_values and sub_criteria. They are mutually exclusive. \n accepted_values: {}.\n sub_criteria: {}".format(self.accepted_values, self.sub_criteria)) 
        
        # Initialize empty list if sub_criteria is None
        # if self.sub_criteria is None:
        #     self.sub_criteria = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert criterion to dictionary format."""
        result = {
            "name": self.name,
            "description": self.description,
        }
        if self.sub_criteria is not None:
            result["sub_criteria"] = [sub.to_dict() for sub in self.sub_criteria]
        else:
            result["sub_criteria"] = []
        
        # Only include accepted_values if not None (SubCritic removes them when sub_criteria exist)
        if self.accepted_values is not None:
            result["accepted_values"] = self.accepted_values 
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Criterion':
        """Create criterion from dictionary."""
        sub_criteria = []
        if "sub_criteria" in data:
            sub_criteria = [cls.from_dict(sub) for sub in data.get("sub_criteria", [])]

        return cls(
            name=data["name"],
            description=data["description"],
            accepted_values=data.get("accepted_values"),  # Can be None 
            sub_criteria=sub_criteria
        )
    
    @staticmethod
    def parse_json_str(json_str: str) -> List['Criterion']:
        """Parse criteria from JSON string."""
        try:
            data = json.loads(json_str)
            return [Criterion.from_dict(item) for item in data]
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid criteria JSON: {e}")
    
    @staticmethod
    def write_json(criteria: List['Criterion']) -> str:
        """Convert criteria list to JSON string."""
        data = [criterion.to_dict() for criterion in criteria]
        return json.dumps(data, indent=2)

    @staticmethod
    def from_criteria_list(criteria_list) -> List[Criterion]:
        """Parse CriteriaList (with sub_criteria) into List of Criterion."""
        result = []
        
        for eval_criterion in criteria_list.criteria:
            sub_criteria = []
            
            # Handle sub_criteria if they exist
            if eval_criterion.sub_criteria:
                # New structure: sub_criteria is a list of EvaluationCriterion objects
                for sub_eval_criterion in eval_criterion.sub_criteria:
                    sub_criterion = Criterion(
                        name=sub_eval_criterion.name,
                        description=sub_eval_criterion.description,
                        accepted_values=sub_eval_criterion.accepted_values,
                        sub_criteria=None  # Sub-criteria should not have nested sub-criteria
                    )
                    sub_criteria.append(sub_criterion)
            
            # Create main criterion
            criterion = Criterion(
                name=eval_criterion.name,
                description=eval_criterion.description,
                accepted_values=eval_criterion.accepted_values if not sub_criteria else None,
                sub_criteria=sub_criteria if sub_criteria else None
            )
            
            result.append(criterion)
        
        return result


# @dataclass
# class EvaluationResult:
#     """Results from a complete evaluation."""
#     workflow_id: str
#     task: Task
#     test_case: str
#     ground_truth: Optional[str]
#     criteria: List[Criterion]
#     quantification_results: Dict[str, Any]
#     final_score: Optional[float]
#     errors: List[str]
#     raw_responses: Dict[str, str]
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert result to dictionary format."""
#         return {
#             "workflow_id": self.workflow_id,
#             "task": {
#                 "name": self.task.name,
#                 "description": self.task.description,
#                 "expected_output": self.task.expected_output
#             },
#             "test_case": self.test_case,
#             "ground_truth": self.ground_truth,
#             "criteria": [criterion.to_dict() for criterion in self.criteria],
#             "quantification_results": self.quantification_results,
#             "final_score": self.final_score,
#             "errors": self.errors,
#             "raw_responses": self.raw_responses
#         }
    
#     def summary(self) -> str:
#         """Generate a summary of the evaluation results."""
#         summary = f"Evaluation Results for: {self.task.name}\n"
#         summary += f"Workflow ID: {self.workflow_id}\n"
#         summary += f"Criteria Count: {len(self.criteria)}\n"
        
#         if self.final_score is not None:
#             summary += f"Final Score: {self.final_score}/10\n"
        
#         if self.errors:
#             summary += f"Errors: {len(self.errors)}\n"
        
#         return summary
