"""
Critic Agent for LangGraph-based AgentEval framework.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from .base_agent import BaseAgent


class SubCriterionDetails(BaseModel):
    """Details for a single sub-criterion."""
    name: str = Field(..., description="Name of the sub-criterion")
    description: str = Field(..., description="Description of what this sub-criterion evaluates")
    accepted_values: List[str] = Field(..., description="Possible accepted inputs for this sub-criterion")

class EvaluationCriterion(BaseModel):
    """Single evaluation criterion with name, description, and accepted values."""
    name: str = Field(..., description="Name of the criterion")
    description: str = Field(..., description="Description of what this criterion evaluates")
    accepted_values: Optional[List[str]] = Field(None, description="List of possible accepted inputs for this criterion")
    sub_criteria: Optional[List['EvaluationCriterion']] = Field(None, description="List of sub-criteria for this criterion")

class CriteriaList(BaseModel):
    """List of evaluation criteria."""
    criteria: List[EvaluationCriterion] = Field(..., description="List of evaluation criteria")

# Rebuild the model to handle forward references
EvaluationCriterion.model_rebuild()
CriteriaList.model_rebuild()

class CriticAgent(BaseAgent):
    """Agent that generates evaluation criteria for tasks.
    Example output:
    [
        {
            "name": "criterion_name",
            "description": "What this criterion evaluates",
            "accepted_values": ["Value1", "Value2", "Value3"]
        }
    ]
    """
    
    DEFAULT_SYSTEM_MESSAGE = """
    You are a helpful assistant. You suggest criteria for evaluating different tasks. They should be distinguishable, quantifiable and not redundant.
    Do not include any criteria that needs to be assessed with external data or information not provided in the task context.
    
    For each criterion, provide:
    - A clear, descriptive name
    - A detailed description of what the criterion evaluates
    - A list of possibly accepted input values for this criterion that are fine-grained and preferably multi-graded levels

    Return the criteria in the specified structured format.
    """
    
    def __init__(
        self,
        name: str = "critic",
        system_message: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        llm: Optional[BaseChatModel] = None,
        **kwargs
    ):
        # Create parser for structured output (fallback)
        self.parser = PydanticOutputParser(pydantic_object=CriteriaList)
        
        # Update system message to include format instructions
        base_system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE
        enhanced_system_message = f"{base_system_message}\n\n{self.parser.get_format_instructions()}"
        
        super().__init__(
            name=name,
            system_message=enhanced_system_message,
            llm_config=llm_config or {},
            description="Agent that generates evaluation criteria for tasks",
            llm=llm,
            **kwargs
        )
    
    def generate_criteria(self, task_message: str, additional_instructions: str = "") -> CriteriaList:
        """Generate evaluation criteria for a given task."""
        prompt = task_message
        if additional_instructions:
            prompt += f"\n\nAdditional Instructions: {additional_instructions}"
        
        prompt += "\n\nPlease generate comprehensive evaluation criteria for this task."
        
        response = self.invoke(prompt)
        return self.parser.parse(response)
    
    async def agenerate_criteria(self, task_message: str, additional_instructions: str = "") -> CriteriaList:
        """Async generate evaluation criteria for a given task."""
        prompt = task_message
        if additional_instructions:
            prompt += f"\n\nAdditional Instructions: {additional_instructions}"
        
        prompt += "\n\nPlease generate comprehensive evaluation criteria for this task."
        
        response = await self.ainvoke(prompt)
        return self.parser.parse(response)