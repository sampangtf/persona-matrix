#!/usr/bin/env python3
"""
Dimension Shift Workflow using LangGraph
"""

from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, START, END
import logging
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter

from .dimension_shift_agents import ExtractorAgent, TransformerAgent, ValidatorAgent, DimensionConfig

logger = logging.getLogger(__name__)

class ShiftState(TypedDict):
    original_summary: str      # immutable Level 0 text
    previous_summary: str      # N-1 level text to be refined  
    current_summary: str       # working draft of N level text
    dimension_config: DimensionConfig
    attempt: int
    original_metadata: Dict[str, Any]    # Metrics for original_summary (from Extractor)
    current_metadata: Dict[str, Any]     # Metrics for current_summary (from Validator)
    validation_result: Dict[str, Any]

class DimensionShiftWorkflow:
    """LangGraph workflow for dimension shifting with LangChain rate limiting"""
    
    def __init__(self, llm=None, llm_config: Dict[str, Any] = None, run_name: str = "Dimension Shift Workflow"):
        """Initialize with either a pre-configured LLM or llm_config for backward compatibility"""
        
        logger.info(f"Initializing DimensionShiftWorkflow with run_name: {run_name}")
        
        if llm is not None:
            # Use pre-configured LLM (preferred approach)
            logger.info("Using pre-configured LLM instance")
            self.llm = llm
            self.llm_config = llm_config or {}
        else:
            # Fallback: create LLM from config (backward compatibility)
            logger.info("Creating LLM from config (backward compatibility mode)")
            self.llm_config = llm_config or {
                "model": "gemini-2.0-flash-lite", 
                "temperature": 0.1, 
                "rate_limit_rpm": 15
            }
            logger.info(f"LLM config: model={self.llm_config.get('model')}, temperature={self.llm_config.get('temperature')}, rate_limit_rpm={self.llm_config.get('rate_limit_rpm')}")
            
            # Create rate limiter using LangChain's InMemoryRateLimiter
            rpm_limit = self.llm_config.get("rate_limit_rpm", 15)
            requests_per_second = rpm_limit / 60.0
            logger.info(f"Setting up rate limiter: {rpm_limit} RPM ({requests_per_second:.3f} RPS)")
            self.rate_limiter = InMemoryRateLimiter(
                requests_per_second=requests_per_second,
                check_every_n_seconds=1,
                max_bucket_size=rpm_limit
            )
            
            # Create shared rate-limited LLM instance
            api_key = self.llm_config.get("google_api_key") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("Google API key not found. Please set google_api_key in llm_config or GOOGLE_API_KEY environment variable.")
                raise ValueError("Google API key not found. Set google_api_key in llm_config or GOOGLE_API_KEY environment variable.")
                
            logger.info("Creating ChatGoogleGenerativeAI instance with rate limiting")
            self.llm = ChatGoogleGenerativeAI(
                model=self.llm_config.get("model", "gemini-2.0-flash-lite"),
                temperature=self.llm_config.get("temperature", 0.1),
                google_api_key=api_key,
                rate_limiter=self.rate_limiter  # Apply rate limiter to LLM
            )
        
        # Adjust temperature for specific agents
        self.transformer_llm = self.llm.model_copy()
        self.transformer_llm.temperature = 0.5

        self.validator_llm = self.llm.model_copy()
        self.validator_llm.temperature = 0.0

        # Initialize agents with shared LLM
        logger.info("Initializing agents (ExtractorAgent, TransformerAgent, ValidatorAgent)")
        self.extractor = ExtractorAgent()
        self.transformer = TransformerAgent(llm=self.transformer_llm)
        self.validator = ValidatorAgent(llm=self.validator_llm)
        
        logger.info("Building LangGraph workflow")
        self.graph = self._build_graph().with_config({"run_name": run_name})
        logger.info("DimensionShiftWorkflow initialization complete")
    
    def _build_graph(self) -> StateGraph:
        logger.info("Building LangGraph state graph with nodes: extractor, transformer, validator")
        workflow = StateGraph(ShiftState)
        
        workflow.add_node("extractor", self._extractor_node)
        workflow.add_node("transformer", self._transformer_node)
        workflow.add_node("validator", self._validator_node)
        
        workflow.add_edge(START, "extractor")
        workflow.add_edge("extractor", "transformer")
        workflow.add_edge("transformer", "validator")
        workflow.add_conditional_edges("validator", self._route, {"retry": "transformer", "end": END})
        
        logger.info("LangGraph state graph compilation complete")
        return workflow.compile()
    
    def _extractor_node(self, state: ShiftState) -> Dict[str, Any]:
        """Extract metadata from original summary (only runs once per dimension)"""
        # Only extract original metadata if not already done
        if not state.get("original_metadata"):
            logger.info("Running extractor to extract metadata from original summary")
            try:
                original_metadata = self.extractor.extract_metadata(state["original_summary"])
                logger.info(f"Metadata extracted successfully: word_count={original_metadata.get('word_count')}, "
                           f"citations={original_metadata.get('citation_count')}, "
                           f"procedural_keywords={original_metadata.get('procedure_keyword_count')}")
                return {"original_metadata": original_metadata}
            except Exception as e:
                logger.error(f"Failed to extract metadata from original summary: {e}")
                raise
        else:
            logger.info("Original metadata already extracted, skipping extractor")
        return {}
    
    def _transformer_node(self, state: ShiftState) -> Dict[str, Any]:
        """Transform previous summary to current level"""
        # Retrieve failure reason from previous validation, if any, to inform retry
        failure_reason = state.get("validation_result", {}).get("failure_reason")
        attempt_num = state["attempt"] + 1
        
        if failure_reason:
            logger.warning(f"Transformer retry attempt #{attempt_num} - Previous failure: {failure_reason}")
        else:
            logger.info(f"Transformer attempt #{attempt_num} for {state['dimension_config'].dimension} level {state['dimension_config'].level}")
        
        try:
            transformed = self.transformer.transform_summary(
                state["previous_summary"],
                state["original_summary"],
                state["dimension_config"],
                state.get("original_metadata"),
                failure_reason
            )
            
            word_count = len(transformed.split()) if transformed else 0
            logger.info(f"Transformation complete - Generated text with {word_count} words")
            
            return {
                "current_summary": transformed,
                "attempt": attempt_num,
                # expose failure_reason for downstream or logging
                "failure_reason": failure_reason
            }
        except Exception as e:
            logger.error(f"Transformer failed on attempt #{attempt_num}: {e}")
            raise
    
    def _validator_node(self, state: ShiftState) -> Dict[str, Any]:
        """Validate transformation with hard checks and LLM critique"""
        logger.info(f"Running validation for {state['dimension_config'].dimension} level {state['dimension_config'].level} attempt #{state['attempt']}")
        
        try:
            result = self.validator.validate_transformation(
                state["current_summary"], 
                state["dimension_config"], 
                state["original_metadata"],
                state["original_summary"],
                state["previous_summary"]
            )
            
            if result["passed"]:
                logger.info("Validation PASSED - transformation meets requirements")
            else:
                logger.warning(f"Validation FAILED - {result.get('failure_reason', 'Unknown reason')}")
                
            return {
                "validation_result": result, 
                "current_metadata": result["new_metadata"]
            }
        except Exception as e:
            logger.error(f"Validator failed during validation: {e}")
            raise
    
    def _route(self, state: ShiftState) -> str:
        """Route based on validation result and attempt count"""
        passed = state["validation_result"]["passed"]
        attempt = state["attempt"]
        max_attempts = 3
        
        if passed:
            logger.info(f"Routing to END - validation passed on attempt #{attempt}")
            return "end"
        elif attempt >= max_attempts:
            logger.warning(f"Routing to END - max attempts ({max_attempts}) reached without success")
            return "end"
        else:
            logger.warning(f"Routing to RETRY - validation failed, attempt #{attempt}/{max_attempts}")
            return "retry"
    
    def shift_summary(self, original_summary: str, previous_summary: str, dimension_config: DimensionConfig) -> Dict[str, Any]:
        """Shift summary from previous level to target level"""
        logger.info(f"Starting shift_summary for {dimension_config.dimension} level {dimension_config.level}")
        
        initial_state = ShiftState(
            original_summary=original_summary,
            previous_summary=previous_summary,
            current_summary="",  # Will be filled by transformer
            dimension_config=dimension_config,
            attempt=0,
            original_metadata={},  # Will be filled by extractor
            current_metadata={},   # Will be filled by validator
            validation_result={}
        )
        
        try:
            result = self.graph.invoke(initial_state)
            
            success = result["validation_result"]["passed"]
            attempts = result["attempt"]
            
            if success:
                logger.info(f"shift_summary SUCCESSFUL for {dimension_config.dimension} level {dimension_config.level} after {attempts} attempts")
            else:
                logger.error(f"shift_summary FAILED for {dimension_config.dimension} level {dimension_config.level} after {attempts} attempts")
            
            return {
                "success": success,
                "transformed_text": result["current_summary"],
                "attempts": attempts,
                "validation_result": result["validation_result"],
                "metadata": result["current_metadata"]
            }
        except Exception as e:
            logger.error(f"shift_summary encountered error for {dimension_config.dimension} level {dimension_config.level}: {e}")
            raise
    
    def multi_level_shift(self, summary_text: str, dimension: str, target_level: int, 
                         dimension_params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform recursive multi-level shift for a dimension"""
        logger.info(f"Starting multi_level_shift for dimension '{dimension}' targeting level {target_level}")
        
        # Extract metadata for level 0 (original summary)
        original_metadata = self.extractor.extract_metadata(summary_text)
        
        results = {
            "dimension": dimension, 
            "levels": {0: {"text": summary_text, "success": True, "metadata": original_metadata}}
        }
        
        current_text = summary_text
        original_text = summary_text
        
        for level in range(1, target_level + 1):
            logger.info(f"Processing {dimension} level {level}/{target_level}")
            
            try:
                # Calculate level-specific parameters
                level_params = self._get_level_params(dimension, level, dimension_params)
                logger.info(f"Level {level} parameters: {level_params}")
                
                # Add previous level metadata for relative threshold validation
                # For level 1, use original metadata as "previous" for comparison
                if level == 1:
                    level_params["previous_metadata"] = original_metadata
                elif level > 1 and "levels" in results and level-1 in results["levels"]:
                    previous_metadata = results["levels"][level-1].get("metadata", {})
                    level_params["previous_metadata"] = previous_metadata
                
                config = DimensionConfig(
                    dimension=dimension, 
                    level=level, 
                    target_params=level_params
                )
                
                # Perform shift from previous level to current level
                result = self.shift_summary(original_text, current_text, config)
                results["levels"][level] = result
                
                if result["success"]:
                    current_text = result["transformed_text"]
                    logger.info(f"Level {level} transformation successful")
                else:
                    # Stop processing if validation fails
                    failure_reason = result.get('validation_result', {}).get('failure_reason', 'Unknown reason')
                    logger.warning(f"Failed to transform {dimension} to level {level}: {failure_reason}")
                    logger.warning(f"Stopping multi-level transformation at level {level}")
                    break
                    
            except Exception as e:
                logger.error(f"Error processing {dimension} level {level}: {e}")
                results["levels"][level] = {
                    "success": False,
                    "error": str(e),
                    "transformed_text": current_text,  # Keep previous level text
                    "attempts": 0
                }
                logger.warning(f"Stopping multi-level transformation due to error at level {level}")
                break
        
        successful_levels = sum(1 for level_data in results["levels"].values() if level_data.get("success", False))
        total_levels = len(results["levels"])
        logger.info(f"multi_level_shift complete for {dimension}: {successful_levels}/{total_levels} levels successful")
        
        return results
    
    def _get_level_params(self, dimension: str, level: int, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get level-specific parameters based on dimension and level"""
        
        if dimension == "Depth":  # Depth
            word_targets = {1: 85, 2: 70, 3: 50, 4: 30}
            params = {"word_target_percent": word_targets.get(level, 100)}
            
        elif dimension == "Precision":  # Precision  
            # Readability scores use relative thresholds (lower than previous level)
            # Citation targets remain absolute percentages with 20% tolerance
            # grade_range_target kept as reference for transformer prompts
            grade_range_target = {1: (12, 13), 2: (11, 12), 3: (9, 11), 4: (7, 9)}
            ncr_score_range_targets = {1: (8.5, 9.0), 2: (8.0, 8.5), 3: (7.0, 8.0), 4: (6.0, 7.0)}
            citation_targets = {1: 75, 2: 50, 3: 25, 4: 5}
            
            params = {
                "citation_target_percent": citation_targets.get(level, 100),
                "grade_target": grade_range_target.get(level, (13, 25)),  # Reference for transformer
                "ncr_score_range_targets": ncr_score_range_targets.get(level, (9.0, 50.0)),  # Reference for transformer
                "level": level
            }
            
            
        elif dimension == "Procedural":  # Procedural
            # Procedural keyword frequency uses relative thresholds (lower than previous level)
            # keyword_target_percent kept as reference for transformer prompts
            keyword_targets = {1: 75, 2: 50, 3: 25, 4: 5}
            
            params = {
                "keyword_target_percent": keyword_targets.get(level, 100),  # Reference for transformer
                "level": level
            }
            
        else:
            params = base_params
            
        logger.debug(f"Generated level parameters for {dimension} level {level}: {params}")
        return params
