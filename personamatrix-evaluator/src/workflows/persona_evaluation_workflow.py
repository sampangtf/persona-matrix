"""
Persona-based evaluation workflow using LangGraph for agent orchestration.
Extends the base evaluation workflow to support persona-specific criteria generation.
"""
from typing import Dict, Any, List, TypedDict, Literal, Optional
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command
from langgraph.checkpoint.memory import MemorySaver
from langsmith import traceable
import operator
import json
import uuid
import logging
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
 
from .evaluation_workflow import parse_criteria_response
from ..models.state import Task, Criterion
from ..models.persona import Persona
from ..agents import PersonaCriticAgent, QuantifierAgent, SubCriticAgent
from ..utils.batch_collector import BatchRequestCollector, create_batch_request_key, BatchInterrupt

logger = logging.getLogger(__name__)


def rpm_to_rate_limiter(rpm_limit: int = 15) -> InMemoryRateLimiter:
    """
    Create a rate limiter based on RPM (requests per minute).
    """ 

    requests_per_second = rpm_limit / 60.0
    logger.info(f"Setting up rate limiter: {rpm_limit} RPM ({requests_per_second:.3f} RPS)")
    
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=0.5,
        max_bucket_size=rpm_limit # max tokens to be accumulated = rpm_limit
    )
    return rate_limiter

# Reducers
def merge_dicts(left: Dict, right: Dict) -> Dict:
    """Merge two dictionaries, with right taking precedence."""
    return {**left, **right} 

def keep_left(left, right):
    """Keep the left value, ignoring the right."""
    return left

def keep_right(left, right):
    """Keep the right value, ignoring the left."""
    return right

class PersonaCriteriaGenerationState(TypedDict, total=False):
    """State for persona-based criteria generation workflow."""
    workflow_id: str 
    thread_id: str
    task: Task
    personas: Annotated[List[Persona], keep_right]
    llm: Optional[BaseChatModel]
    llm_config: Dict[str, Any]
    additional_instructions: str
    max_round: int
    use_subcritic: bool
    
    # # Branch-local "cursor" (set per Send branch)
    # current_persona_index: Annotated[List[int], operator.add]  # Track all indices as a list
    # # Branch-local round robin state
    # round_counts: Annotated[Dict[str, int], merge_dicts]
    # current_steps: Annotated[Dict[str, str], merge_dicts]
    
    # Collect errors across branches
    errors: Annotated[List[str], operator.add]
    
    # Final outputs (merged only at gather_results)
    persona_criteria: Annotated[Dict[str, List[Criterion]], merge_dicts]
    persona_raw_responses: Annotated[Dict[str, str], merge_dicts]
    persona_refined_responses: Annotated[Dict[str, str], merge_dicts]


class PersonaBranchState(TypedDict, total=False):
    """Isolated branch state for individual persona processing."""
    persona_index: int
    persona: Persona
    task: Task
    llm: Optional[BaseChatModel]
    llm_config: Dict[str, Any]
    additional_instructions: str
    max_round: int
    use_subcritic: bool
    
    # Branch-local state (isolated from other branches)
    current_round: int
    current_step: str  # "critic" or "subcritic"
    criteria: List[Criterion]
    raw_response: str
    refined_response: str
    error: str

# === PERSONA EVALUATION WORKFLOW ===
class PersonaQuantificationState(TypedDict, total=False):
    """State for persona-based quantification workflow."""
    workflow_id: str
    thread_id: str
    task: Task
    persona_criteria: Dict[str, List[Criterion]]  # persona_name -> criteria
    llm: Optional[BaseChatModel]
    llm_config: Dict[str, Any]
    test_case: str
    ground_truth: str
    
    # Batch collection metadata (required for batch mode)
    test_case_id: str
    dimension: str
    level: int
    
    # Batch collection mode
    batch_mode: bool
    batch_collector: BatchRequestCollector
    
    # Collect errors across branches
    errors: Annotated[List[str], operator.add]
    
    # Outputs (merged across branches using reducers)
    persona_quantification_results: Annotated[Dict[str, Dict[str, Any]], merge_dicts]
    persona_raw_quantification_responses: Annotated[Dict[str, str], merge_dicts]


# === PERSONA CRITERIA GENERATION WORKFLOW ===

def initialize_persona_criteria_generation(state: PersonaCriteriaGenerationState) -> PersonaCriteriaGenerationState:
    """Initialize persona-based criteria generation workflow."""
    
    # if use_subcritic is enabled, double the max_round for round-robin
    # This allows for one critic round and one subcritic round per user input round 
 

    return {
        "workflow_id": state.get("workflow_id") or str(uuid.uuid4()),
        "thread_id": state.get("thread_id"),
        "task": state.get("task"),
        "personas": state.get("personas", []),
        "llm": state.get("llm"),
        "llm_config": state.get("llm_config"),
        "additional_instructions": state.get("additional_instructions", ""),
        "max_round": state.get("max_round", 1),
        "use_subcritic": state.get("use_subcritic", False),
        "errors": state.get("errors", []),
        "persona_criteria": {},
        "persona_raw_responses": {},
        "persona_refined_responses": {},
    }


def dispatch_personas(state: PersonaCriteriaGenerationState):
    """Dispatch personas to isolated branches using Send API."""
    personas = state.get("personas")
    if not personas:
        logger.error("No personas to dispatch, ending workflow")
        return END
    
    logger.info(f"Dispatching {len(personas)} personas to isolated branches using Send API") 
    
    parent_graph_llm = state.get("llm")
    if hasattr(parent_graph_llm, "rate_limiter"):
        original_rps_limit = parent_graph_llm.rate_limiter.requests_per_second 
        rpm_limit = original_rps_limit * 60  # Convert RPS to RPM

        branch_rps = original_rps_limit / len(personas)
        branch_rate_limiter = InMemoryRateLimiter(
            requests_per_second=branch_rps,
            check_every_n_seconds=0.1,
            max_bucket_size=rpm_limit # max tokens to be accumulated = rpm_limit
        )
        dispatched_llm = parent_graph_llm.model_copy()
        dispatched_llm.rate_limiter = branch_rate_limiter
    else:
        dispatched_llm = parent_graph_llm or state.get("llm_config", {}).get("model")
    
    # Create isolated branch for each persona with Send API directed to the subgraph
    return [Send("persona_branch_subgraph", {
        "persona_index": i,
        "persona": persona,
        "task": state.get("task"),
        "llm": dispatched_llm,
        "llm_config": state.get("llm_config"),
        "additional_instructions": state.get("additional_instructions", ""),
        "max_round": state.get("max_round", 1),
        "use_subcritic": state.get("use_subcritic", False),
        "current_round": 0,
        "current_step": "critic",
        "criteria": [],
        "raw_response": "",
        "refined_response": "",
        "error": ""
    }) for i, persona in enumerate(personas)]


def persona_branch_critic(state: PersonaBranchState) -> PersonaBranchState:
    """Run the persona critic agent in isolated branch."""
    persona = state["persona"]
    persona_name = persona.name
    current_round = state.get("current_round", 0) + 1
    
    # Update state for critic step
    updated_state = {
        **state,
        "current_round": current_round,
        "current_step": "critic", 
    }
    
    try:
        # Initialize persona critic agent with additional instructions 
        agent = PersonaCriticAgent(
            persona=persona, 
            llm_config=state.get("llm_config"),
            llm=state.get("llm")
        )
        
        # Generate criteria
        task_message = state["task"].get_sys_message()
        additional_instructions = state.get("additional_instructions", "")

        # If this is not the first round, provide previous criteria as context
        existing_criteria = state.get("criteria", [])
        if current_round > 1 and existing_criteria:
            previous_criteria = [c.to_dict() for c in existing_criteria]
            previous_criteria_json = json.dumps(previous_criteria, indent=2)
            additional_instructions += (
                "\n\nHere are the criteria generated in the previous round:\n"
                f"{previous_criteria_json}\n"
                "Please review, improve, and refine these criteria. "
                "You may add, remove, or reword criteria for clarity, completeness, and relevance. "
                "If you see duplicates or unclear items, merge or clarify them. "
                "Return the full revised list."
            )

        response = agent.generate_criteria(task_message, additional_instructions=additional_instructions)
        
        # Parse to Criterion list
        criteria: List[Criterion] = []
        
        try:
            parsed = parse_criteria_response(response)
            criteria = parsed or []
            logger.info(f"Generated {len(criteria)} criteria for persona '{persona_name}' (round {current_round})")
        except (json.JSONDecodeError, ValueError) as e:
            error_msg = f"Failed to parse criteria for {persona_name}: {e}"
            logger.exception(error_msg)
            # ERROR RETURN: JSON/parsing error - unable to parse LLM response into valid criteria format
            return {
                **updated_state,
                "raw_response": str(response),
                "error": error_msg
            }

        # Update state with successful results
        return {
            **updated_state,
            "raw_response": str(response),
            "criteria": criteria,
            "error": ""  # Clear error if successful
        }

    except Exception as e:
        error_msg = f"Persona critic failed for {persona_name}: {e}"
        logger.exception(error_msg)
        # ERROR RETURN: General exception - agent initialization, LLM call, or other unexpected failure
        return {
            **updated_state, 
            "error": error_msg
        }


def persona_branch_subcritic(state: PersonaBranchState) -> PersonaBranchState:
    """Run the persona subcritic agent in isolated branch."""
    persona = state["persona"]
    persona_name = persona.name
    current_round = state.get("current_round", 0) + 1

    # Update state for subcritic step
    updated_state = {
        **state,
        "current_round": current_round,
        "current_step": "subcritic", 
    }

    existing_criteria = state.get("criteria", [])
    if not existing_criteria:
        logger.warning(f"No criteria to refine for persona '{persona_name}', skipping subcritic step")
        # EARLY RETURN: No criteria available - skip subcritic processing (not an error)
        return {
            **updated_state,
            "current_round": current_round,
            "current_step": "subcritic"
        }
    
    try:
        # Initialize subcritic agent
        subcritic = SubCriticAgent(llm_config=state["llm_config"], llm=state.get("llm"))
        
        # Prepare criteria for refinement
        criteria_dicts = [criterion.to_dict() for criterion in existing_criteria]
        existing_criteria_json = json.dumps(criteria_dicts, indent=2)
        
        # Refine criteria
        task_message = state["task"].get_sys_message()
        response = subcritic.refine_criteria(task_message, existing_criteria_json)
        
        # Parse refined criteria
        try:
            refined_criteria = parse_criteria_response(response)
            if refined_criteria:
                logger.info(f"Refined criteria for persona '{persona_name}' with {len(refined_criteria)} items (round {current_round})")
                
                # Update state with successful results
                return {
                    **updated_state,
                    "criteria": refined_criteria,
                    "refined_response": str(response),
                    "error": ""
                }
            else:
                logger.warning(f"No valid refined criteria found for persona '{persona_name}', keeping original")
                # PARTIAL SUCCESS RETURN: LLM responded but no valid criteria parsed - keep original criteria
                return {
                    **updated_state,
                    "raw_response": str(response),
                    "refined_response": str(response),
                    "error": ""
                }
        
        except (json.JSONDecodeError, ValueError) as e:
            error_msg = f"Error parsing refined criteria for {persona_name}, keeping original: {e}"
            logger.warning(error_msg)
            # ERROR RETURN: JSON/parsing error - unable to parse refined criteria, fallback to original
            return {
                **updated_state,
                "raw_response": str(response),
                "refined_response": str(response),
                "error": error_msg
            }
    
    except Exception as e:
        error_msg = f"Subcritic agent failed for {persona_name}: {e}"
        logger.error(error_msg)
        # ERROR RETURN: General exception - agent initialization, LLM call, or other unexpected failure
        return {
            **updated_state,
            "error": error_msg
        }

def persona_branch_router(state: PersonaBranchState) -> Literal["persona_branch_critic", "persona_branch_subcritic", "gather_results"]:
    """Route between critic, subcritic, and gather based on current state and configuration."""
    current_round = state.get("current_round", 0)
    max_round = state.get("max_round", 1)
    use_subcritic = state.get("use_subcritic", False)
    current_step = state.get("current_step", "critic")
    error = state.get("error", "")
    
    # If there's an error, go to gather results
    if error:
        return "gather_results"
    
    # If max rounds reached, go to gather results
    if current_round >= max_round:
        return "gather_results"
    
    # If subcritic is enabled and we just finished a critic step, go to subcritic
    if use_subcritic and current_step == "critic":
        return "persona_branch_subcritic"
    
    # If we just finished subcritic or subcritic is not enabled, continue with critic
    if current_step == "subcritic" or not use_subcritic:
        return "persona_branch_critic"
    
    # Default fallback
    return "gather_results"


def gather_persona_results(state: PersonaBranchState) -> Command:
    """Gather results from isolated persona branch back to main state using Command to parent graph."""
    persona = state["persona"]
    persona_name = persona.name
    
    # Extract final results from isolated branch
    criteria = state.get("criteria", [])
    raw_response = state.get("raw_response", "")
    refined_response = state.get("refined_response", "")
    error = state.get("error", "")
    
    logger.info(f"Gathering results for persona '{persona_name}': {len(criteria)} criteria, error='{error}'")
    
    # Prepare results for merging back to main state
    result = {}
    
    if criteria:
        result["persona_criteria"] = {persona_name: criteria}
    
    if raw_response:
        result["persona_raw_responses"] = {persona_name: raw_response}
    
    if refined_response:
        result["persona_refined_responses"] = {persona_name: refined_response}
    
    if error:
        result["errors"] = [f"Persona {persona_name}: {error}"]
    
    # Use Command to return to parent graph and update main state
    return Command(graph=Command.PARENT, update=result, goto="finalize_results")


def finalize_persona_criteria_generation(state: PersonaCriteriaGenerationState) -> PersonaCriteriaGenerationState:
    """Finalize persona-based criteria generation."""
    workflow_id = state.get("workflow_id", "unknown")
    total_criteria = sum(len(criteria) for criteria in state.get("persona_criteria", {}).values())
    personas_count = len(state.get("personas", []))
    
    logger.info(f"Persona criteria generation workflow {workflow_id} completed with {total_criteria} total criteria across {personas_count} personas")
    return {}


def create_persona_branch_subgraph() -> StateGraph:
    """Create a subgraph for persona branch processing."""
    subgraph_builder = StateGraph(PersonaBranchState)
    
    # Add nodes for the subgraph
    subgraph_builder.add_node("persona_branch_critic", persona_branch_critic)
    subgraph_builder.add_node("persona_branch_subcritic", persona_branch_subcritic)
    subgraph_builder.add_node("gather_results", gather_persona_results)
    
    # Entry point to the subgraph
    subgraph_builder.add_edge(START, "persona_branch_critic")
    
    # Router to determine next step
    subgraph_builder.add_conditional_edges(
        "persona_branch_critic", 
        persona_branch_router,
        {
            "persona_branch_critic": "persona_branch_critic",
            "persona_branch_subcritic": "persona_branch_subcritic", 
            "gather_results": "gather_results"
        }
    )
    
    subgraph_builder.add_conditional_edges(
        "persona_branch_subcritic",
        persona_branch_router,
        {
            "persona_branch_critic": "persona_branch_critic",
            "persona_branch_subcritic": "persona_branch_subcritic",
            "gather_results": "gather_results"
        }
    )
    
    # gather_results uses Command to exit back to parent, so no explicit edge to END
    
    return subgraph_builder.compile().with_config({"run_name": "Persona Branch Subgraph"})


def create_persona_criteria_generation_workflow() -> StateGraph:
    """Create a persona-based criteria generation workflow using Send API for isolated branches."""
    builder = StateGraph(PersonaCriteriaGenerationState)

    # Create the persona branch subgraph
    persona_subgraph = create_persona_branch_subgraph()

    # Main workflow nodes
    builder.add_node("initialize", initialize_persona_criteria_generation)
    builder.add_node("persona_branch_subgraph", persona_subgraph)  # Use subgraph as a node
    builder.add_node("finalize_results", finalize_persona_criteria_generation)

    # Main workflow edges
    builder.add_edge(START, "initialize")
    builder.add_conditional_edges("initialize", dispatch_personas, ["persona_branch_subgraph", END])
    
    # finalize_results is reached via Command from gather_results in subgraph
    builder.add_edge("finalize_results", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory).with_config({"run_name": "Persona Criteria Generation Workflow"})


# === PERSONA QUANTIFICATION WORKFLOW ===

def initialize_persona_quantification(state: PersonaQuantificationState) -> PersonaQuantificationState:
    """Initialize persona-based quantification workflow."""
    return {
        "workflow_id": state.get("workflow_id") or str(uuid.uuid4()),
        "thread_id": state.get("thread_id"),
        "task": state.get("task"),
        "persona_criteria": state.get("persona_criteria"),
        "llm": state.get("llm"),
        "llm_config": state.get("llm_config"),
        "test_case": state.get("test_case"),
        "ground_truth": state.get("ground_truth", ""),
        "batch_mode": state.get("batch_mode", False),
        "batch_collector": state.get("batch_collector"),
        "errors": state.get("errors", []),
        "persona_quantification_results": {},
        "persona_raw_quantification_responses": {},
    }


def dispatch_quantification(state: PersonaQuantificationState):
    """Dispatch persona quantification to parallel processing using Send API."""
    persona_criteria = state.get("persona_criteria", {})
    if not persona_criteria:
        logger.info("No persona criteria to quantify, ending workflow")
        return END
    
    parent_graph_llm = state.get("llm")
    if hasattr(parent_graph_llm, "rate_limiter"):
        rpm_limit = parent_graph_llm.rate_limiter.requests_per_second * 60
        dispatched_rpm_limit = rpm_limit / len(persona_criteria)
        dispatched_llm = parent_graph_llm.model_copy()
        dispatched_llm.rate_limiter = rpm_to_rate_limiter(dispatched_rpm_limit)
    else:
        dispatched_llm = parent_graph_llm or state.get("llm_config", {}).get("model")
    
    # Fan out: one branch per persona, each with its own branch-local persona name
    logger.info(f"Dispatching {len(persona_criteria)} persona quantifiers to parallel branches using Send API")
    return [Send("persona_quantifier", {
        "current_persona_name": persona_name,
        "task": state.get("task"),
        "llm": parent_graph_llm,
        "llm_config": state.get("llm_config"),
        "persona_criteria": state.get("persona_criteria", {}),
        "test_case": state.get("test_case"),
        "ground_truth": state.get("ground_truth", ""),
        "batch_mode": state.get("batch_mode", False),
        "batch_collector": state.get("batch_collector"),
        # Include common state values needed for batch request key generation
        "workflow_id": state.get("workflow_id"),
        "test_case_id": state.get("test_case_id"),
        "dimension": state.get("dimension"),
        "level": state.get("level"),
    }) for persona_name in persona_criteria.keys()]


def persona_quantifier(state: PersonaQuantificationState) -> PersonaQuantificationState:
    """Process a single persona quantifier using branch-local state."""
    persona_name = state["current_persona_name"]
    persona_criteria = state.get("persona_criteria", {})
    current_criteria = persona_criteria.get(persona_name, [])
    
    if not current_criteria:
        logger.warning(f"No criteria available for persona '{persona_name}', skipping quantification")
        return {"persona_quantification_results": {persona_name: {"warning": "No criteria available"}}}
    
    try:
        # Initialize quantifier agent
        quantifier = QuantifierAgent(llm_config=state["llm_config"], llm=state.get("llm"))
        
        # Prepare criteria for quantification
        criteria_dicts = [criterion.to_dict() for criterion in current_criteria]
        criteria_json = json.dumps(criteria_dicts, indent=2)
        
        # Quantify performance for this persona
        task_message = state["task"].get_sys_message()
        test_case = state["test_case"]
        ground_truth = state.get("ground_truth", "")
        
        # Add persona context to the quantification
        persona_context = f"\n\nEvaluating from the perspective of: {persona_name}"
        task_message_with_persona = task_message + persona_context
        
        # Check if running in batch mode
        if state.get("batch_mode", False):
            # Collect request for batch processing instead of executing
            batch_collector = state.get("batch_collector")
            if not batch_collector:
                batch_collector = BatchRequestCollector(
                    llm_config=state.get("llm_config")
                )
                state["batch_collector"] = batch_collector
            
            # Get the batch request data
            request_data = quantifier.collect_batch_request(
                task_message_with_persona, criteria_json, test_case, ground_truth, batch_collector
            )
            
            # Create hierarchical key for this persona request
            request_key = create_batch_request_key(
                workflow_id=state["workflow_id"],
                persona_name=persona_name,
                step="quantify",
                test_case_id=state["test_case_id"],
                dimension=state["dimension"],
                level=state["level"],
                evaluation_type="persona_eval"
            )
            
            # Add to batch collector with metadata
            request_metadata = {
                "evaluation_type": "persona_eval",
                "persona_name": persona_name,
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
            
            logger.info(f"Collected quantification request for persona '{persona_name}': {request_key}")
            
            return {
                "persona_quantification_results": {
                    persona_name: {
                        "batch_mode": True,
                        "request_key": request_key,
                        "message": f"Request collected for batch processing"
                    }
                }
            }
        
        else:
            # Normal execution mode - invoke LLM immediately
            response = quantifier.quantify_performance(
                task_message_with_persona, criteria_json, test_case, ground_truth
            )
            
            # Parse quantification results
            try: 
                # Handle structured response
                if hasattr(response, 'model_dump'):
                    results = response.model_dump()
                elif isinstance(response, dict):
                    results = response
                elif isinstance(response, str):
                    # Try to extract JSON from response
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    
                    if json_start != -1 and json_end != 0:
                        results_json = response[json_start:json_end]
                        results = json.loads(results_json)
                    else:
                        raise ValueError("No valid JSON found in quantification response") 
                    
                logger.info(f"Successfully quantified performance for persona '{persona_name}'")
                
                return {
                    "persona_raw_quantification_responses": {persona_name: str(response)},
                    "persona_quantification_results": {persona_name: results},
                }
            
            except (json.JSONDecodeError, ValueError) as e:
                msg = f"Failed to parse quantification results for {persona_name}: {e}"
                logger.exception(msg)
                return {
                    "errors": [msg],
                    "persona_raw_quantification_responses": {persona_name: str(response)},
                    "persona_quantification_results": {persona_name: {"raw_response": str(response), "parsing_error": str(e)}},
                }
    
    except Exception as e:
        msg = f"Quantification failed for {persona_name}: {e}"
        logger.exception(msg)
        return {
            "errors": [msg],
            "persona_quantification_results": {persona_name: {"error": str(e)}},
        }


def gather_quantification_results(state: PersonaQuantificationState) -> PersonaQuantificationState:
    """Gather quantification results from all parallel persona processing."""
    # Purely for logging/metrics; outputs are already merged via reducers
    personas_count = len(state.get("persona_quantification_results", {}))
    logger.info(f"Gathered quantification results from {personas_count} persona branches")
    return {}


def finalize_persona_quantification(state: PersonaQuantificationState) -> PersonaQuantificationState:
    """Finalize persona-based quantification workflow."""
    workflow_id = state.get("workflow_id", "unknown")
    personas_count = len(state.get("persona_quantification_results", {}))
    logger.info(f"Persona quantification workflow {workflow_id} completed for {personas_count} personas")
    return {}


def create_persona_quantification_workflow() -> StateGraph:
    """Create a persona-based quantification workflow using Send API for MapReduce."""
    builder = StateGraph(PersonaQuantificationState)
    
    builder.add_node("initialize", initialize_persona_quantification)
    builder.add_node("persona_quantifier", persona_quantifier)
    # Defer fan-in: run only once all persona_quantifier branches finish
    builder.add_node("gather_quantification_results", gather_quantification_results, defer=True)
    builder.add_node("finalize", finalize_persona_quantification)
    
    builder.add_edge(START, "initialize")
    builder.add_conditional_edges("initialize", dispatch_quantification, ["persona_quantifier", END])
    builder.add_edge("persona_quantifier", "gather_quantification_results")
    builder.add_edge("gather_quantification_results", "finalize")
    builder.add_edge("finalize", END)
    
    memory = MemorySaver()
    return builder.compile(checkpointer=memory).with_config({"run_name": "Persona Quantification Workflow"})


def create_persona_quantification_workflow_no_checkpointing() -> StateGraph:
    """Create a persona-based quantification workflow without checkpointing (for batch mode)."""
    builder = StateGraph(PersonaQuantificationState)
    
    builder.add_node("initialize", initialize_persona_quantification)
    builder.add_node("persona_quantifier", persona_quantifier)
    # Defer fan-in: run only once all persona_quantifier branches finish
    builder.add_node("gather_quantification_results", gather_quantification_results, defer=True)
    builder.add_node("finalize", finalize_persona_quantification)
    
    builder.add_edge(START, "initialize")
    builder.add_conditional_edges("initialize", dispatch_quantification, ["persona_quantifier", END])
    builder.add_edge("persona_quantifier", "gather_quantification_results")
    builder.add_edge("gather_quantification_results", "finalize")
    builder.add_edge("finalize", END)
    
    # Compile without checkpointer for batch mode (to avoid serialization issues)
    return builder.compile().with_config({"run_name": "Persona Quantification Workflow (Batch Mode)"})
