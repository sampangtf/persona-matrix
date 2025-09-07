"""
Base agent class for LangGraph-based AgentEval framework.
"""
from typing import Dict, Any, Optional, Union, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
import logging

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base agent class with LangGraph integration."""
    
    def __init__(
        self,
        name: str,
        system_message: str,
        llm_config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        llm: Optional[BaseChatModel] = None
    ):
        self.name = name
        self.system_message = system_message
        self.description = description or system_message
        
        # Use provided LLM instance or create from config
        if llm is not None:
            self.llm = llm
            self.llm_config = llm_config  # Keep for reference but don't use to create LLM
        elif llm_config is not None:
            self.llm_config = llm_config
            self.llm = self._create_llm_from_config(llm_config)
        else:
            raise ValueError("Either llm_config or llm instance must be provided")
    
    def _create_llm_from_config(self, llm_config: Dict[str, Any]) -> BaseChatModel:
        """Create LangChain LLM from configuration.""" 
        model_name = llm_config.get("model", "gemini-2.0-flash-lite")

        temperature = llm_config.get("temperature", 0.7)
        
        # Handle Gemini models
        if "gemini" in model_name.lower():
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                api_key = llm_config.get("api_key") or os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not found in environment or config")
                
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=api_key,
                    **{k: v for k, v in llm_config.items() if k not in ["model", "temperature", "api_key"]}
                )
            except ImportError:
                logger.error("langchain_google_genai not available")
                raise
        
        # Handle Claude models
        elif "claude" in model_name.lower():
            try:
                from langchain_anthropic import ChatAnthropic
                api_key = llm_config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not found in environment or config")
                
                return ChatAnthropic(
                    model=model_name,
                    temperature=temperature,
                    anthropic_api_key=api_key,
                    **{k: v for k, v in llm_config.items() if k not in ["model", "temperature", "api_key"]}
                )
            except ImportError:
                logger.error("langchain_anthropic not available")
                raise
        
        # Handle OpenAI models
        else:
            try:
                from langchain_openai import ChatOpenAI
                api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment or config")
                
                return ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    openai_api_key=api_key,
                    **{k: v for k, v in llm_config.items() if k not in ["model", "temperature", "api_key"]}
                )
            except ImportError:
                logger.error("langchain_openai not available")
                raise
    
    def invoke(self, message: str, **kwargs) -> str:
        """Invoke the LLM with a message."""
        try:
            messages = [
                SystemMessage(content=self.system_message),
                HumanMessage(content=message)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content
            
            # Handle new OpenAI structured multi-modal content format
            if isinstance(content, list) and len(content) > 0:
                # Extract text from the first content block 
                first_block = content[0]
                for block in content:
                    if block['type'] =='text':
                        return block['text'] 
                    
            # Fall back to string conversion if content format is unexpected
            return str(content) if content is not None else ""
        except Exception as e:
            logger.error(f"Error invoking {self.name}: {e}")
            raise
    
    async def ainvoke(self, message: str, **kwargs) -> str:
        """Async invoke the LLM with a message."""
        try:
            messages = [
                SystemMessage(content=self.system_message),
                HumanMessage(content=message)
            ]
            
            response = await self.llm.ainvoke(messages)
            content = response.content
            
            # Handle new OpenAI structured content format
            if isinstance(content, list) and len(content) > 0:
                # Extract text from the first content block
                first_block = content[0]
                if isinstance(first_block, dict) and 'text' in first_block:
                    return first_block['text']
                elif isinstance(first_block, dict) and 'type' in first_block and first_block['type'] == 'text':
                    return first_block.get('text', str(content))
            
            # Fall back to string conversion if content format is unexpected
            return str(content) if content is not None else ""
        except Exception as e:
            logger.error(f"Error async invoking {self.name}: {e}")
            raise