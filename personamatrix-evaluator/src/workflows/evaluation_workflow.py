"""
Core evaluation workflow using LangGraph for agent orchestration.
Provides decoupled functions for criteria generation and quantification.
Supports batch request collection for Google Gemini Batch API.
"""
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import json
import uuid
import logging
from langchain.output_parsers import PydanticOutputParser
from langchain_core.language_models import BaseChatModel

from ..models.state import Task, Criterion
from ..agents import CriticAgent, QuantifierAgent, SubCriticAgent, CriteriaList, QuantificationResult
from ..utils.batch_collector import BatchRequestCollector, create_batch_request_key, BatchInterrupt

logger = logging.getLogger(__name__)

critic_agent_parser = PydanticOutputParser(pydantic_object=CriteriaList)   

def parse_criteria_response(response: Any) -> List[Criterion]:
    """Parse criteria response from CriticAgent."""  

    # If response is a Pydantic model, use its dict method 
    if isinstance(response, CriteriaList):
        criteria_list = response
        
    # If response is a string, try to extract JSON  
    elif isinstance(response, str):
        logger.warning("Response is a string but not expected CriteriaList, attempting to parse JSON...")
        # Try to extract JSON from response
        # This assumes the response contains a JSON array of criteria
        json_start = response.find("[")
        json_end = response.rfind("]") + 1
        
        if json_start != -1 and json_end != 0:
            criteria_json = response[json_start:json_end]
            criteria_list = critic_agent_parser.parse(criteria_json)
        else:
            logger.error("No valid JSON found in response") 
            return []
    else:
        logger.error(f"Unexpected response type: {type(response)}")
        return []
 
    # Convert from CriteriaList to List[Criterion] objects
    try: 
        # If response is already a CriteriaList, convert it directly
        criteria = Criterion.from_criteria_list(criteria_list) 
    except Exception as e:
        logger.error(f"Error converting CriteriaList to List[Criterion]: {e}. Response: {criteria_list}")
        return []
 
    return criteria
    

class CriteriaGenerationState(TypedDict, total=False):
    """State for criteria generation workflow."""
    workflow_id: str 
    thread_id: str
    task: Task
    llm: Optional[BaseChatModel]
    llm_config: Dict[str, Any]
    additional_instructions: str
    max_round: int
    use_subcritic: bool
    round_count: int
    current_step: str
    errors: List[str]
    
    # Generated data
    criteria: List[Criterion]
    raw_criteria_response: str
    refined_criteria_response: str


class QuantificationState(TypedDict, total=False):
    """State for quantification workflow."""
    workflow_id: str
    thread_id: str
    task: Task
    criteria: List[Criterion]
    llm: Optional[BaseChatModel]
    llm_config: Dict[str, Any]
    test_case: str
    ground_truth: str
    current_step: str
    errors: List[str]
    
    # Batch collection metadata (required for batch mode)
    test_case_id: str
    dimension: str 
    level: int
    
    # Batch collection mode
    batch_mode: bool
    batch_collector: BatchRequestCollector
    
    # Generated data
    quantification_results: Dict[str, Any]
    raw_quantification_response: str


# === CRITERIA GENERATION WORKFLOW ===

def initialize_criteria_generation(state: CriteriaGenerationState) -> CriteriaGenerationState:
    """Initialize criteria generation workflow."""
    if not state.get("workflow_id"):
        state["workflow_id"] = str(uuid.uuid4()) 
    
    state["current_step"] = "initialize"
    state["errors"] = state.get("errors", [])
    state["round_count"] = 0
    
    logger.info(f"Initialized criteria generation workflow {state['workflow_id']}")
    return state


def critic_agent_step(state: CriteriaGenerationState) -> CriteriaGenerationState:
    """Run the critic agent to generate initial criteria."""
    state["current_step"] = "critic"
    state["round_count"] += 1
    
    try:
        # Initialize critic agent with additional instructions
        # system_message = CriticAgent.DEFAULT_SYSTEM_MESSAGE
        # if state.get("additional_instructions"):
        #     system_message += f"\n{state['additional_instructions']}"
        
        critic = CriticAgent(
            # system_message=system_message,
            llm_config=state["llm_config"],
            llm=state.get("llm")
        )
        
        # Generate criteria
        task_message = state["task"].get_sys_message()
        additional_instructions = state.get("additional_instructions", "")

        # If this is not the first round, provide previous criteria as context
        if state["round_count"] > 1 and state.get("criteria"):
            previous_criteria = [c.to_dict() for c in state["criteria"]]
            previous_criteria_json = json.dumps(previous_criteria, indent=2)
            additional_instructions += (
            "\n\nHere are the criteria generated in the previous round:\n"
            f"{previous_criteria_json}\n"
            "Please review, improve, and refine these criteria. "
            "You may add, remove, or reword criteria for clarity, completeness, and relevance. "
            "If you see duplicates or unclear items, merge or clarify them. "
            "Return the full revised list."
            )

        response = critic.generate_criteria(
            task_message,
            additional_instructions=additional_instructions
        )
        state["raw_criteria_response"] = response
        # Parse criteria from response
        try:
            criteria = parse_criteria_response(response)
            state["criteria"] = criteria
            logger.info(f"Generated {len(criteria)} criteria")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing criteria JSON: {e}")
            state["errors"].append(f"Failed to parse criteria: {str(e)}. Model Response: {response}")
             
            state["criteria"] = []
    
    except Exception as e:
        logger.error(f"Error in critic agent step: {e}")
        state["errors"].append(f"Critic agent failed: {str(e)}.")
        state["criteria"] = []
    
    return state


def subcritic_agent_step(state: CriteriaGenerationState) -> CriteriaGenerationState:
    """Run the subcritic agent to refine criteria."""
    state["current_step"] = "subcritic"
    state["round_count"] += 1
    
    if not state.get("criteria"):
        logger.warning("No criteria to refine, skipping subcritic step")
        return state
    
    try:
        # Initialize subcritic agent
        subcritic = SubCriticAgent(llm_config=state["llm_config"], llm=state.get("llm"))
        
        # Prepare criteria for refinement
        criteria_dicts = [criterion.to_dict() for criterion in state["criteria"]]
        existing_criteria = json.dumps(criteria_dicts, indent=2)
        
        # Refine criteria
        task_message = state["task"].get_sys_message()
        response = subcritic.refine_criteria(task_message, existing_criteria)
        state["refined_criteria_response"] = response
        
        # Parse refined criteria
        try:
            refined_criteria = parse_criteria_response(response)
            if refined_criteria:
                state["criteria"] = refined_criteria
                logger.info(f"Refined criteria with {len(refined_criteria)} items")  

            else:
                logger.warning("No valid refined criteria found, keeping original. Response: {response}")
        
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Error parsing refined criteria, keeping original: {e}. Response: {response}")
    
    except Exception as e:
        logger.error(f"Error in subcritic agent step: {e}")
        state["errors"].append(f"Subcritic agent failed: {str(e)}")
    
    return state


def should_continue_criteria_generation(state: CriteriaGenerationState) -> str:
    """Determine if criteria generation should continue."""
    # Check if we've reached max rounds
    if state["round_count"] >= state["max_round"]:
        return "finalize"
    
    # If we have more rounds available, determine next step based on current step
    current_step = state.get("current_step", "")
    use_subcritic = state.get("use_subcritic", False)
    
    # After critic step: go to subcritic if enabled, otherwise check for more rounds
    if current_step == "critic":
        if use_subcritic:
            return "subcritic"
        else:
            # No subcritic, check if we need more critic rounds
            return "critic" if state["round_count"] < state["max_round"] else "finalize"
    
    # After subcritic step: always go back to critic for round-robin
    elif current_step == "subcritic":
        return "critic" if state["round_count"] < state["max_round"] else "finalize"
    
    # Default fallback
    return "finalize"


def should_use_subcritic(state: CriteriaGenerationState) -> str:
    """Determine if subcritic should be used after critic step."""
    use_subcritic = state.get("use_subcritic", False)
    
    if use_subcritic:
        return "subcritic"
    else:
        # Check if more rounds are needed
        if state["round_count"] < state["max_round"]:
            return "more_rounds"
        else:
            return "finalize"


def check_more_rounds(state: CriteriaGenerationState) -> str:
    """Check if more rounds are needed when subcritic is not used."""
    if state["round_count"] < state["max_round"]:
        return "critic"
    else:
        return "finalize"


def finalize_criteria_generation(state: CriteriaGenerationState) -> CriteriaGenerationState:
    """Finalize criteria generation."""
    state["current_step"] = "finalize"
    logger.info(f"Criteria generation workflow {state['workflow_id']} completed with {len(state.get('criteria', []))} criteria")
    return state


def create_criteria_generation_workflow() -> StateGraph:
    """Create the criteria generation workflow."""
    workflow = StateGraph(CriteriaGenerationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_criteria_generation)
    workflow.add_node("critic", critic_agent_step)
    workflow.add_node("subcritic", subcritic_agent_step)
    workflow.add_node("finalize", finalize_criteria_generation)
    
    # Add edges following the mermaid diagram
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "critic")
    
    # After critic: check if subcritic should be used
    workflow.add_conditional_edges(
        "critic",
        should_use_subcritic,
        {
            "subcritic": "subcritic",
            "more_rounds": "critic",
            "finalize": "finalize"
        }
    )
    
    # After subcritic: check if more rounds are needed (always goes back to critic in round-robin)
    workflow.add_conditional_edges(
        "subcritic", 
        should_continue_criteria_generation,
        {
            "critic": "critic",
            "finalize": "finalize"
        }
    )
    
    workflow.add_edge("finalize", END)
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory).with_config({"run_name": "Criterion Generation Workflow"})


# === QUANTIFICATION WORKFLOW ===

def initialize_quantification(state: QuantificationState) -> QuantificationState:
    """Initialize quantification workflow."""
    if not state.get("workflow_id"):
        state["workflow_id"] = str(uuid.uuid4()) 

    state["current_step"] = "initialize"
    state["errors"] = state.get("errors", [])
    
    logger.info(f"Initialized quantification workflow {state['workflow_id']}")
    return state


def quantifier_agent_step(state: QuantificationState) -> QuantificationState:
    """Run the quantifier agent to quantify performance or collect for batch processing."""
    state["current_step"] = "quantify"
    
    if not state.get("criteria"):
        error_msg = "No criteria available for quantification"
        logger.error(error_msg)
        state["errors"].append(error_msg)
        return state
    
    try:
        # Initialize quantifier agent
        quantifier = QuantifierAgent(llm_config=state["llm_config"], llm=state.get("llm"))
        
        # Prepare criteria for quantification
        criteria_dicts = [criterion.to_dict() for criterion in state["criteria"]]
        criteria_json = json.dumps(criteria_dicts, indent=2)
        
        # Get parameters
        task_message = state["task"].get_sys_message()
        test_case = state["test_case"]
        ground_truth = state.get("ground_truth", "")
        
        # Check if running in batch mode
        if state.get("batch_mode", False):
            # Collect request for batch processing instead of executing
            batch_collector = state.get("batch_collector")
            if not batch_collector:
                error_msg = "Batch mode requires a batch_collector to be provided"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                state["quantification_results"] = {"error": error_msg}
                return state
            
            # Get the batch request data
            request_data = quantifier.collect_batch_request(
                task_message, criteria_json, test_case, ground_truth, batch_collector
            )
            
            # Create hierarchical key for this request
            request_key = create_batch_request_key(
                workflow_id=state["workflow_id"],
                step="quantify", 
                test_case_id=state["test_case_id"],
                dimension=state["dimension"],
                level=state["level"],
                evaluation_type="agent_eval"
            )
            
            # Add to batch collector with metadata
            request_metadata = {
                "evaluation_type": "agent_eval",
                "dimension": state["dimension"],
                "level": state["level"],
                "test_case_id": state["test_case_id"]
            }
            
            batch_collector.add_request(
                request_key,
                request_data["messages"],
                request_data["config"],
                request_metadata
            )
            
            logger.info(f"Collected quantification request for batch processing: {request_key}")
            
            # Store batch collection info in state (don't save/interrupt here - let caller handle it)
            state["quantification_results"] = {
                "batch_mode": True,
                "request_key": request_key,
                "request_added": True,
                "message": "Request collected for batch processing"
            }
        
        else:
            # Normal execution mode - invoke LLM immediately
            response = quantifier.quantify_performance(
                task_message, criteria_json, test_case, ground_truth
            )
            
            state["raw_quantification_response"] = str(response)
            
            # Parse quantification results
            try: 
                # If response is a Pydantic model, use its dict method
                if isinstance(response, QuantificationResult):
                    state["quantification_results"] = response.model_dump()
                
                # The quantifier should return structured data, but let's handle string responses too
                elif isinstance(response, str):
                    print("Response is a string but not expected QuantificationResult, attempting to parse JSON...")
                    # Try to extract JSON from response
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    
                    if json_start != -1 and json_end != 0:
                        results_json = response[json_start:json_end]
                        results = json.loads(results_json)
                        state["quantification_results"] = results
                    else:
                        raise ValueError("No valid JSON found in quantification response") 
                    
                logger.info("Successfully quantified performance")
            
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing quantification results: {e}")
                logger.error(f"Raw response: {response}")

                # Store raw response and error
                state["errors"].append(f"Failed to parse quantification results: {str(e)}")
                
                # Create basic results structure
                state["quantification_results"] = {
                    "raw_response": str(response),
                    "parsing_error": str(e)
                }
    
    except BatchInterrupt:
        # Re-raise batch interrupts to stop workflow
        raise
    except Exception as e:
        logger.error(f"Error in quantification: {e}")
        state["errors"].append(f"Quantification failed: {str(e)}")
        state["quantification_results"] = {"error": str(e)}
    
    return state


def finalize_quantification(state: QuantificationState) -> QuantificationState:
    """Finalize quantification workflow."""
    state["current_step"] = "finalize"
    logger.info(f"Quantification workflow {state['workflow_id']} completed")
    return state


def create_quantification_workflow() -> StateGraph:
    """Create the quantification workflow."""
    workflow = StateGraph(QuantificationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_quantification)
    workflow.add_node("quantify", quantifier_agent_step)
    workflow.add_node("finalize", finalize_quantification)
    
    # Add edges
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "quantify")
    workflow.add_edge("quantify", "finalize")
    workflow.add_edge("finalize", END)
    
    # Compile with memory
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory).with_config({"run_name": "Quantification Workflow"})


def create_quantification_workflow_no_checkpointing() -> StateGraph:
    """Create the quantification workflow without checkpointing (for batch mode)."""
    workflow = StateGraph(QuantificationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_quantification)
    workflow.add_node("quantify", quantifier_agent_step)
    workflow.add_node("finalize", finalize_quantification)
    
    # Add edges
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "quantify")
    workflow.add_edge("quantify", "finalize")
    workflow.add_edge("finalize", END)
    
    # Compile without checkpointer for batch mode (to avoid serialization issues)
    return workflow.compile().with_config({"run_name": "Quantification Workflow (Batch Mode)"})