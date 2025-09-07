"""
SubCritic Agent for LangGraph-based AgentEval framework.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from .base_agent import BaseAgent
from .critic_agent import EvaluationCriterion, CriteriaList, SubCriterionDetails

# class SubCriterionDetails(BaseModel):
#     """Details for a single sub-criterion."""
#     description: str = Field(..., description="Description of what this sub-criterion evaluates")
#     accepted_values: List[str] = Field(..., description="Possible accepted inputs for this sub-criterion")


# class RefinedCriterion(BaseModel):
#     """A refined criterion that may include sub-criteria."""
#     name: str = Field(..., description="Name of the criterion")
#     description: str = Field(..., description="Description of what this criterion evaluates")
#     sub_criteria: Optional[Dict[str, SubCriterionDetails]] = Field(None, description="Dictionary mapping sub-criterion names to their details")


# class RefinedCriteriaList(BaseModel):
#     """List of refined evaluation criteria."""
#     criteria: List[RefinedCriterion] = Field(..., description="List of refined evaluation criteria")


class SubCriticAgent(BaseAgent):
    """Agent that refines and adds sub-criteria to existing evaluation criteria.
    Example output:
    [
        {
            "name": "criterion_name",
            "description": "What this criterion evaluates",  
            "sub_criteria": {
                "sub_criterion_name": {
                    "description": "What this sub-criterion evaluates",
                    "accepted_values": ["SubValue1", "SubValue2"]
                }
            }

        }
    ]
    """
    
    DEFAULT_SYSTEM_MESSAGE = """
    You are a helpful assistant to the critic agent. You suggest sub criteria for evaluating different tasks based on the criteria provided by the critic agent (if you feel it is needed).
    They should be distinguishable, quantifiable, and related to the overall theme of the critic's provided criteria.
    Your role is to:
    - Analyze existing criteria descriptipn and determine if sub-criteria would improve evaluation granularity 
    - For each criterion that benefits from sub-criteria, remove the original "accepted_values" and replace with detailed sub-criteria
    - For each sub-criterion, provide clear descriptions and a list of possibly accepted input values for this sub-criterion that are fine-grained and preferably multi-graded levels
    
    Return the refined criteria in the specified structured format.
    """
    
    def __init__(
        self,
        name: str = "subcritic",
        system_message: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        llm: Optional[BaseChatModel] = None,
        **kwargs
    ):
        # Create parser for structured output (fallback)
        self.parser = PydanticOutputParser(pydantic_object=CriteriaList)
        
        # Update system message to include format instructions
        base_system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE
        enhanced_system_message = f"{base_system_message}\n\n{self.parser.get_format_instructions()}."
        
        super().__init__(
            name=name,
            system_message=enhanced_system_message,
            llm_config=llm_config or {},
            llm=llm,
            description="Agent that refines and adds sub-criteria to evaluation criteria"
        )
    
    def refine_criteria(
        self,
        task_message: str,
        existing_criteria: str,
        additional_instructions: str = ""
    ) -> CriteriaList:
        """Refine existing criteria by adding sub-criteria and improvements."""
        prompt = f"Task Context:\n{task_message}\n\n"
        prompt += f"Existing Criteria to Refine:\n{existing_criteria}\n\n"
        
        if additional_instructions:
            prompt += f"Additional Instructions: {additional_instructions}\n\n"
        
        prompt += "Please refine these criteria by adding relevant sub-criteria and ensuring comprehensive evaluation coverage."
        
        response = self.invoke(prompt)
        return self.parser.parse(response)
    
    async def arefine_criteria(
        self,
        task_message: str,
        existing_criteria: str,
        additional_instructions: str = ""
    ) -> CriteriaList:
        """Async refine existing criteria by adding sub-criteria and improvements."""
        prompt = f"Task Context:\n{task_message}\n\n"
        prompt += f"Existing Criteria to Refine:\n{existing_criteria}\n\n"
        
        if additional_instructions:
            prompt += f"Additional Instructions: {additional_instructions}\n\n"
        
        prompt += "Please refine these criteria by adding relevant sub-criteria and ensuring comprehensive evaluation coverage."
        
        response = await self.ainvoke(prompt)
        return self.parser.parse(response)