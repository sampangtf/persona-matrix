"""
Quantifier Agent for LangGraph-based AgentEval framework.
"""
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from .base_agent import BaseAgent
import logging

class CriteriaResult(BaseModel):
    """Result for a criterion evaluation."""
    performance: str = Field(..., description="Assessed performance based on accepted values")
    numerical_score: float = Field(..., ge=0.0, le=5.0, description="Numerical score (0-5)")


class QuantificationResult(BaseModel):
    """
    Result of quantifying performance based on evaluation criteria.

    The 'evaluations' field maps each criterion name to either:
      - a CriteriaResult (for flat criteria), or
      - a dictionary mapping sub-criterion names directly to CriteriaResult (for criteria with sub-criteria).

    For criteria with sub-criteria, the value should be a dict where each key is the sub-criterion name and the value is a CriteriaResult.

    Example:
    {
        "evaluations": {
            "criteria_name1": {
                "performance": "assessed performance based on accepted values for each criteria",
                "numerical_score": 1.5
            },
            "criteria_name2": {
                "sub_criteria_name1": {
                    "performance": "assessed performance for sub-criteria",
                    "numerical_score": 5.0
                },
                "sub_criteria_name2": {
                    "performance": "assessed performance for sub-criteria", 
                    "numerical_score": 2.0
                }
            }
        }
    }
    """
    evaluations: Dict[str, Union[CriteriaResult, Dict[str, CriteriaResult]]] = Field(
        ..., 
        description=(
            "Dictionary mapping criterion names to either a CriteriaResult (for flat criteria) "
            "or a dictionary of sub-criterion names to CriteriaResult (for criteria with sub-criteria). "
        )
    )


class QuantifierAgent(BaseAgent):
    """Agent that quantifies performance based on evaluation criteria.
    Example output:
    
    {"evaluations":
        {
            "criteria_name1": {
                "performance": "assessed performance based on accepted values for each criteria",
                "numerical_score": 2.5
            },
            "criteria_name2": {
                "sub_criteria_name1": {
                    "performance": "assessed performance for sub-criteria",
                    "numerical_score": 5.0
                },
                "sub_criteria_name2": {
                    "performance": "assessed performance for sub-criteria", 
                    "numerical_score": 3.0
                }
            }
        }
    }
    """
    
    DEFAULT_SYSTEM_MESSAGE = """
    You are a helpful assistant. You quantify the output of different tasks based on the given criteria.
    The criterion is given in a json list format where each element is a distinct criterion.
    For each criterion provided, evaluate the test case and provide:
    - If a criterion has direct "accepted_values": provide assessed performance based on "accepted values" for each criterion and numerical_score (0-5)
    - If a criterion has "sub_criteria": evaluate each sub-criterion individually with performance and numerical_score
    
    Output format:
    - For flat criteria: {"performance": "assessment", "numerical_score": score}
    - For criteria with sub-criteria: {"sub_criterion_name": {"performance": "assessment", "numerical_score": score}, ...}
    
    Return your evaluation in the specified structured format with precise numerical scores.
    """.strip()
    
    def __init__(
        self,
        name: str = "quantifier",
        system_message: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        llm: Optional[BaseChatModel] = None,
        **kwargs
    ):
        # Create parser for structured output
        self.parser = PydanticOutputParser(pydantic_object=QuantificationResult)
        
        # Update system message to include format instructions
        base_system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE
        enhanced_system_message = f"{base_system_message}\n\n{self.parser.get_format_instructions()}"
        
        super().__init__(
            name=name,
            system_message=enhanced_system_message,
            llm_config=llm_config or {},
            llm=llm,
            description="Agent that quantifies performance based on evaluation criteria"
        )
        
        # Create retry parser for handling validation errors
        self.retry_parser = RetryOutputParser.from_llm(parser=self.parser, llm=self.llm)
    
    def _build_quantification_prompt(
        self,
        task_message: str,
        criteria_json: str,
        test_case: str,
        ground_truth: str = ""
    ) -> str:
        """Build the quantification prompt with consistent formatting."""

        prompt = f"{task_message}"
        prompt += f"Evaluation Criteria: {criteria_json}"
        prompt += f"Test Case to Evaluate: {test_case}"
        
        if ground_truth:
            prompt += f"Ground Truth/Expected Result: {ground_truth}"
        
        prompt += "Please evaluate the test case against each criterion and provide your assessment."
        
        return prompt

    def _parse_with_retry(self, response: str, prompt: str) -> QuantificationResult:
        """Parse response with retry on validation errors."""
        try:
            return self.parser.parse(response)
        except Exception as e:
            logging.warning(f"Initial parsing failed: {e}. Attempting retry...")
            # For retry, we need to provide both the failed response and original prompt
            # The retry parser will ask the LLM to fix the response
            from langchain_core.prompt_values import StringPromptValue
            prompt_value = StringPromptValue(text=prompt)
            return self.retry_parser.parse_with_prompt(response, prompt_value)

    def quantify_performance(
        self,
        task_message: str,
        criteria_json: str,
        test_case: str,
        ground_truth: str = ""
    ) -> QuantificationResult:
        """Quantify performance based on criteria and test case."""
        prompt = self._build_quantification_prompt(task_message, criteria_json, test_case, ground_truth)
        
        response = self.invoke(prompt)
        return self._parse_with_retry(response, prompt)
    
    def collect_batch_request(
        self,
        task_message: str,
        criteria_json: str,
        test_case: str,
        ground_truth: str = "",
        batch_collector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Collect request for batch processing instead of immediate execution.
        
        Returns:
            Dict containing the request information for batch collection
        """
        logging.info(f"QuantifierAgent: Starting batch request collection")
        logging.debug(f"QuantifierAgent: batch_collector type: {type(batch_collector)}")
        logging.debug(f"QuantifierAgent: batch_collector provider: {batch_collector.provider if batch_collector else 'None'}")
        
        prompt = self._build_quantification_prompt(task_message, criteria_json, test_case, ground_truth)
        
        provider = batch_collector.provider if batch_collector else "gemini"
        logging.info(f"QuantifierAgent: provider={provider}")
        
        logging.debug(f"QuantifierAgent: Using system message with format instructions")
        # Create messages in the appropriate format for each provider
        if provider == "openai":
            # OpenAI format uses role-based messages
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt}
            ]
        else:
            # Other providers may use different format
            messages = [
                {"type": "system", "content": self.system_message},
                {"type": "human", "content": prompt}
            ]
        
        # Configure request based on provider - using basic JSON response format
        if provider == "openai":
            config = {
                "response_format": {
                    "type": "json_object"
                }
            }
            logging.info(f"QuantifierAgent: Setting OpenAI config with JSON object response format")
        else:
            # Use Gemini format or other providers
            config = { 
                "response_mime_type": "application/json"
            }
            logging.info(f"QuantifierAgent: Setting {provider} config with response_mime_type")
            logging.debug(f"QuantifierAgent: Config: {config}")
        
        result = {
            "messages": messages,
            "config": config,
            "prompt": prompt
        }
        
        logging.info(f"QuantifierAgent: Batch request collection completed")
        logging.debug(f"QuantifierAgent: Returning result with keys: {list(result.keys())}")
        
        return result
    
    async def aquantify_performance(
        self,
        task_message: str,
        criteria_json: str,
        test_case: str,
        ground_truth: str = ""
    ) -> QuantificationResult:
        """Async quantify performance based on criteria and test case."""
        prompt = self._build_quantification_prompt(task_message, criteria_json, test_case, ground_truth) 
        
        response = await self.ainvoke(prompt)
        return self._parse_with_retry(response, prompt)