"""
Unified batch request collection for evaluation workflows.
Combines the new minimal BatchCollector with the interface expected by existing workflows.
"""
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal
import re

logger = logging.getLogger(__name__)

ProviderType = Literal["gemini", "openai", "anthropic"]


class BatchRequestCollector:
    """
    Multi-provider batch request collector with custom_id support.
    
    Collects LLM requests for batch processing instead of immediate execution.
    Supports Gemini, OpenAI, and Anthropic APIs with proper custom_id handling
    for request identification and result tracking.
    
    Key Features:
    - Custom ID support for all providers (Gemini, OpenAI, Anthropic)
    - Hierarchical request keys for evaluation framework identification
    - Provider-specific request formatting
    - Enhanced metadata tracking with request breakdowns
    - Test case grouping and organization
    
    Custom ID Format:
    - Each request gets a custom_id that identifies the evaluation framework
    - Format: case_{id}_{eval_type}_dim_{dimension}_level_{level}_persona_{name}
    - Enables precise tracking of which evaluation framework made each call
    """
    
    def __init__(self, provider: ProviderType = "gemini", output_dir: str = "batch_requests", test_case_id: Optional[str] = None, llm_config: Optional[Dict[str, Any]] = None):
        self.provider = provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.requests = []
        self.session_id = str(uuid.uuid4())[:8]
        self.test_case_id = test_case_id
        self.llm_config = llm_config or {}  # Store default LLM config
        
        # Enhanced metadata tracking
        self.metadata = {
            "session_id": self.session_id,
            "test_case_id": test_case_id,
            "created_at": datetime.now().isoformat(),
            "request_breakdown": {
                "agent_eval": 0,
                "persona_eval": 0,
                "by_dimension": {}
            },
            "llm_config": self.llm_config  # Track LLM config in metadata
        }
        
    def add_request(
        self,
        key: str,
        messages: List[Dict[str, str]],
        config: Optional[Dict[str, Any]] = None,
        request_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a request to the batch collection with proper custom_id support."""
        logger.info(f"BatchCollector: Adding request for key: {key}, provider: {self.provider}")
        logger.debug(f"BatchCollector: Request config: {config}")
        logger.debug(f"BatchCollector: Request metadata: {request_metadata}")
        
        # Parse key to update metadata
        self._update_metadata_from_key(key, request_metadata or {})
        
        # Merge instance LLM config with request-specific config
        merged_config = {**self.llm_config}
        if config:
            merged_config.update(config)
            logger.debug(f"BatchCollector: Merged config: {merged_config}")
        
        # Format request based on provider
        if self.provider == "gemini":
            jsonl_entry = self._format_gemini_request(key, messages, merged_config)
        elif self.provider == "openai":
            jsonl_entry = self._format_openai_request(key, messages, merged_config)
        elif self.provider == "anthropic":
            jsonl_entry = self._format_anthropic_request(key, messages, merged_config)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        self.requests.append(jsonl_entry)
        logger.info(f"BatchCollector: Successfully added {self.provider} batch request with custom_id: {key}")
        logger.debug(f"BatchCollector: Total requests now: {len(self.requests)}")
        return key
        
    def _format_gemini_request(self, key: str, messages: List[Dict[str, str]], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Format request for Gemini Batch API."""
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg.get("type") == "system" or msg.get("role") == "system":
                system_instruction = {"parts": [{"text": msg["content"]}]}
            elif msg.get("type") == "human" or msg.get("role") == "user":
                contents.append({
                    "parts": [{"text": msg["content"]}],
                    "role": "user"
                })
        
        # Build the request in Gemini batch format
        request = {"contents": contents}
        if system_instruction:
            request["systemInstruction"] = system_instruction
        if config:
            request["generationConfig"] = config
            
        # Gemini uses "custom_id" in the root level for batch requests
        return {
            "custom_id": key,
            "request": request
        }
    
    def _format_openai_request(self, key: str, messages: List[Dict[str, str]], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Format request for OpenAI Batch API using the new /v1/responses endpoint."""
        logger.info(f"BatchCollector: Formatting OpenAI request for key: {key}")
        logger.debug(f"BatchCollector: Input config: {config}")
        
        # Convert messages to the new responses API format
        input_messages = []
        
        for msg in messages:
            if msg.get("type") == "system" or msg.get("role") == "system":
                input_messages.append({
                    "role": "system",
                    "content": [{"type": "input_text", "text": msg["content"]}]
                })
            elif msg.get("type") == "human" or msg.get("role") == "user":
                input_messages.append({
                    "role": "user", 
                    "content": [{"type": "input_text", "text": msg["content"]}]
                })
            elif msg.get("role") == "assistant":
                # Convert assistant message to new format
                content = msg.get("content", "")
                if isinstance(content, str):
                    input_messages.append({
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}]
                    })
                else:
                    input_messages.append(msg)
        
        # Build request body for /v1/responses endpoint - only documented parameters
        request_body = {
            "model": config.get("model", "gpt-4o") if config else "gpt-4o",
            "input": input_messages
        }
        
        logger.debug(f"BatchCollector: Base request body: {request_body}")
        
        # Only add parameters that are documented in the Responses API
        if config:
            logger.debug(f"BatchCollector: Processing config parameters: {list(config.keys())}")
            
            # Handle documented parameters only
            if "max_tokens" in config:
                request_body["max_output_tokens"] = config["max_tokens"]
                logger.debug(f"BatchCollector: Set max_output_tokens from max_tokens: {config['max_tokens']}")
            elif "max_output_tokens" in config:
                request_body["max_output_tokens"] = config["max_output_tokens"]
                logger.debug(f"BatchCollector: Set max_output_tokens: {config['max_output_tokens']}")
                
            if "temperature" in config:
                request_body["temperature"] = config["temperature"]
                logger.debug(f"BatchCollector: Set temperature: {config['temperature']}")
                
            if "top_p" in config:
                request_body["top_p"] = config["top_p"]
                logger.debug(f"BatchCollector: Set top_p: {config['top_p']}")
            
            # Handle structured outputs - move response_format to text.format
            if "response_format" in config and isinstance(config["response_format"], dict):
                logger.info(f"BatchCollector: Found response_format in config: {config['response_format']}")
                
                if config["response_format"].get("type") == "json_schema":
                    # For structured outputs with JSON schema
                    text_format = {
                        "type": "json_schema", 
                        "schema": config["response_format"]["schema"]
                    }
                    # Include strict parameter if provided
                    if "strict" in config["response_format"]:
                        text_format["strict"] = config["response_format"]["strict"]
                        logger.debug(f"BatchCollector: Added strict parameter: {config['response_format']['strict']}")
                    
                    request_body["text"] = {"format": text_format}
                    logger.info(f"BatchCollector: ✅ Set text.format with json_schema type and schema")
                    logger.debug(f"BatchCollector: Full text format: {text_format}")
                    
                elif config["response_format"].get("type") == "json_object":
                    # For basic JSON object format
                    request_body["text"] = {
                        "format": {"type": "json_object"}
                    }
                    logger.warning(f"BatchCollector: ⚠️ Using json_object format (no schema) for key: {key}")
                    logger.debug(f"BatchCollector: json_object format set")
                else:
                    logger.warning(f"BatchCollector: Unknown response_format type: {config['response_format'].get('type')}")
            
            # Handle legacy JSON format requests
            elif config.get("response_mime_type") == "application/json":
                # Use basic JSON object format
                request_body["text"] = {
                    "format": {"type": "json_object"}
                }
                logger.warning(f"BatchCollector: ⚠️ Converting response_mime_type=application/json to json_object format (no schema) for key: {key}")
                logger.debug(f"BatchCollector: Legacy format conversion applied")
            else:
                logger.debug(f"BatchCollector: No structured output format specified")
        else:
            logger.debug(f"BatchCollector: No config provided, using defaults")
                
        logger.debug(f"BatchCollector: Final request body: {request_body}")
        
        final_request = {
            "custom_id": key,
            "method": "POST",
            "url": "/v1/responses",  # Updated endpoint
            "body": request_body
        }
        
        logger.info(f"BatchCollector: Completed OpenAI request formatting for key: {key}")
        if "text" in request_body and "format" in request_body["text"]:
            format_type = request_body["text"]["format"].get("type", "unknown")
            has_schema = "schema" in request_body["text"]["format"]
            logger.info(f"BatchCollector: Format type: {format_type}, Has schema: {has_schema}")
        
        return final_request
    
    def _format_anthropic_request(self, key: str, messages: List[Dict[str, str]], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Format request for Anthropic Batch API."""
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if msg.get("type") == "system":
                system_message = msg["content"]
            elif msg.get("type") == "human" or msg.get("role") == "user":
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif msg.get("role") in ["user", "assistant"]:
                anthropic_messages.append(msg)
        
        # Start with required parameters
        request = {
            "model": config.get("model", "claude-3-5-sonnet-20241022") if config else "claude-3-5-sonnet-20241022",
            "max_tokens": config.get("max_tokens", 1024) if config else 1024,
            "messages": anthropic_messages
        }
        
        if system_message:
            request["system"] = system_message
        
        # Pass through ALL config parameters
        if config:
            for config_key, config_value in config.items():
                # Skip parameters that are already handled above
                if config_key in ["model", "max_tokens"]:
                    continue
                # Pass through all other parameters as-is
                request[config_key] = config_value
            
        return {
            "custom_id": key,
            "params": request
        }
        
    def _update_metadata_from_key(self, key: str, request_metadata: Dict[str, Any]):
        """Update metadata based on request key and metadata."""
        # Parse evaluation type from key or metadata
        if "agent_eval" in key:
            self.metadata["request_breakdown"]["agent_eval"] += 1
        elif "persona_eval" in key:
            self.metadata["request_breakdown"]["persona_eval"] += 1
        
        # Parse dimension from key
        if "dim_" in key:
            dim_part = [p for p in key.split("_") if p.startswith("dim_")]
            if dim_part:
                dimension = dim_part[0].replace("dim_", "")
                if dimension not in self.metadata["request_breakdown"]["by_dimension"]:
                    self.metadata["request_breakdown"]["by_dimension"][dimension] = 0
                self.metadata["request_breakdown"]["by_dimension"][dimension] += 1
    
    def save_batch_file(self, filename: Optional[str] = None) -> str:
        """Save batch requests to JSONL file with provider-specific format."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            case_suffix = f"_case_{self.test_case_id}" if self.test_case_id else ""
            filename = f"batch_{self.provider}{case_suffix}_{timestamp}.jsonl"
        
        filepath = self.output_dir / filename
        
        # Update metadata with final stats
        self.metadata.update({
            "total_requests": len(self.requests),
            "provider": self.provider,
            "completed_at": datetime.now().isoformat(),
            "batch_file": str(filepath)
        })
        
        # Save metadata file alongside JSONL
        metadata_file = filepath.with_suffix('.metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save JSONL requests in provider-specific format
        with open(filepath, 'w') as f:
            for request in self.requests:
                f.write(json.dumps(request) + '\n')
        
        logger.info(f"Saved {len(self.requests)} {self.provider} batch requests to: {filepath}")
        logger.info(f"Each request includes custom_id for result tracking")
        logger.info(f"Saved metadata to: {metadata_file}")
        
        return str(filepath)
    
    def clear_requests(self):
        """Clear collected requests."""
        self.requests = []
        
    def get_request_count(self) -> int:
        """Get number of collected requests."""
        return len(self.requests)

def create_batch_request_key(
    workflow_id: str, 
    persona_name: str = None, 
    step: str = "quantify",
    test_case_id: str = None,
    dimension: str = None,
    level: int = None,
    evaluation_type: str = None
    ) -> str:
    """Create a hierarchical key for batch request tracking."""
    parts = []

    # Start with test case if available
    if test_case_id:
        parts.append(f"case_{test_case_id}")

    # Add evaluation type
    if evaluation_type:
        parts.append(evaluation_type)
    elif persona_name:
        parts.append("persona_eval")
    else:
        parts.append("agent_eval")

    # Add dimension and level for dimension shift evaluation
    if dimension:
        parts.append(f"dim_{dimension.lower()}")
    if level is not None:
        parts.append(f"level_{level}")

    # Add persona name for PersonaEval (clean invalid characters)
    if persona_name:
        # Clean persona name: replace spaces and invalid filename characters with underscores
        clean_persona = persona_name.replace(" ", "_").replace("-", "_")
        # Remove or replace other problematic characters
        clean_persona = re.sub(r'[^\w\-_]', '', clean_persona)
        # Ensure it's not empty after cleaning
        if clean_persona:
            parts.append(f"persona_{clean_persona}")

    # Add workflow ID if needed for uniqueness
    if workflow_id and workflow_id not in "_".join(parts):
        parts.append(f"wf_{workflow_id[:8]}")

    # Add step if not default
    if step and step != "quantify":
        parts.append(step)
        
    return "_".join(parts)


class BatchInterrupt(Exception):
    """Exception to interrupt normal LLM execution for batch collection."""
    
    def __init__(self, batch_file: str, request_count: int):
        self.batch_file = batch_file
        self.request_count = request_count
        super().__init__(f"Batch collection interrupt: {request_count} requests saved to {batch_file}")


# Compatibility aliases for the new minimal system
BatchCollector = BatchRequestCollector
create_batch_key = create_batch_request_key


def enable_batch_mode(evaluator, provider: ProviderType = "gemini", case_id: str = None):
    """Enable batch mode for any evaluator."""
    logger.info(f"BatchCollector: Enabling batch mode for evaluator with provider: {provider}, case_id: {case_id}")
    
    # Extract LLM config from evaluator if available
    llm_config = getattr(evaluator, 'llm_config', None)
    logger.debug(f"BatchCollector: Extracted LLM config from evaluator: {llm_config}")
    
    collector = BatchRequestCollector(
        provider=provider, 
        test_case_id=case_id,
        llm_config=llm_config  # Pass the evaluator's LLM config
    )
    evaluator.batch_mode = True
    evaluator.batch_collector = collector
    
    logger.info(f"BatchCollector: Successfully enabled batch mode for evaluator")
    logger.debug(f"BatchCollector: Collector instance created with provider: {collector.provider}")
    
    return collector
