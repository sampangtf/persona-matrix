"""
Persona-lens Critic Agent for PersonaEval framework.
"""
from typing import Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from .critic_agent import CriticAgent, CriteriaList
from ..models.persona import Persona, get_persona
from langchain.output_parsers import PydanticOutputParser


class PersonaCriticAgent(CriticAgent):
    """Critic agent that generates evaluation criteria from a specific persona perspective.
    
    Extends CriticAgent to incorporate persona-specific context and evaluation dimensions
    into the criteria generation process.
    """

    PERSONA_SYSTEM_MESSAGE_TEMPLATE = """
You are a helpful assistant. You suggest criteria for evaluating different tasks from the perspective of {persona_name}. 
They should be distinguishable, quantifiable and not redundant, and reflect the priorities and informational needs of the assigned persona.
Do not include any criteria that needs to be assessed with external data or information not provided in the task context.

{persona_context}

For each criterion, provide:
- A clear, descriptive name
- A detailed description of what the criterion evaluates
- A list of possibly accepted input values for this criterion that are fine-grained and preferably multi-graded levels

Return the criteria in the specified structured format.
"""

    def __init__(
        self,
        persona: Persona,
        name: Optional[str] = None,
        system_message: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        llm: Optional[BaseChatModel] = None,
        **kwargs
    ):
        self.persona = persona
        self.parser  = PydanticOutputParser(pydantic_object=CriteriaList)

        # Use provided system message or create persona-specific one
        if system_message is None:
            system_message = self._create_persona_system_message() 
            enhanced_system_message = f"{system_message}\n\n{self.parser.get_format_instructions()}"  
        
        # Generate default name if not provided
        if name is None:
            name = f"persona_critic_{persona.name.lower().replace(' ', '_').replace('&', 'and')}"

        super().__init__(
            name=name,
            system_message=enhanced_system_message,
            llm_config=llm_config,
            llm=llm,
            **kwargs
        )
    
    def _create_persona_system_message(self) -> str:
        """Create system message incorporating persona context."""
        return self.PERSONA_SYSTEM_MESSAGE_TEMPLATE.format(
            persona_context=self.persona.get_context_prompt(),
            persona_name=self.persona.name.upper()
        )
    
    def _build_persona_prompt(self, task_message: str, additional_instructions: str = "") -> str:
        """Build prompt with persona-specific context."""
        prompt = f"Task Context:\n{task_message}\n\n"
        
        if additional_instructions:
            prompt += f"Additional Instructions: {additional_instructions}"
        
        prompt += f"Please generate comprehensive evaluation criteria for this task from the perspective of {self.persona.name}."
        # prompt += f"\n\n{self.persona.get_context_prompt()}\n\n"
        # prompt += f"Remember to focus on: {', '.join(self.persona.key_evaluation_dimensions)}"
        
        return prompt
    
    def generate_criteria(self, task_message: str, additional_instructions: str = "") -> CriteriaList:
        """Generate evaluation criteria from this persona's perspective.
        
        Overrides base method to include persona-specific context.
        """
        prompt = self._build_persona_prompt(task_message, additional_instructions)
        response = self.invoke(prompt)
        return self.parser.parse(response)
    
    async def agenerate_criteria(self, task_message: str, additional_instructions: str = "") -> CriteriaList:
        """Async generate evaluation criteria from this persona's perspective.
        
        Overrides base method to include persona-specific context.
        """
        prompt = self._build_persona_prompt(task_message, additional_instructions)
        response = await self.ainvoke(prompt)
        return self.parser.parse(response)
    
    def get_persona_info(self) -> Dict[str, Any]:
        """Get information about the assigned persona."""
        return {
            "name": self.persona.name,
            "description": self.persona.description,
            "representative_users": self.persona.representative_users,
            "informational_needs": self.persona.informational_needs,
            "key_evaluation_dimensions": self.persona.key_evaluation_dimensions,
            "mission_alignment": self.persona.mission_alignment
        }


def create_persona_critic_from_name(
    persona_name: str,
    llm_config: Optional[Dict[str, Any]] = None,
    llm: Optional[BaseChatModel] = None,
    **kwargs
) -> PersonaCriticAgent:
    """Create a PersonaCriticAgent from a persona name."""
    persona = get_persona(persona_name)
    if not persona:
        raise ValueError(f"Unknown persona: {persona_name}")
    
    return PersonaCriticAgent(
        persona=persona,
        llm_config=llm_config,
        llm=llm,
        **kwargs
    )