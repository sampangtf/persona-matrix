#!/usr/bin/env python3
"""
Dimension Shift Dataset Generator

This script takes legal case summaries from JSON files and creates dimension shift variants.
Each case gets transformed along 3 dimensions (D, P, R) with 5 levels each (0-4).

The generator follows the same pattern as factual_omission_generator.py:
- Load case data from JSON
- Process each case through the dimension shift workflow
- Create one dimension at a time with recursive level-by-level transformation
- Save results as JSON dataset
"""
import os
import json
import logging
from typing import Dict, List, Any, Literal
from dataclasses import dataclass
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter

from .dimension_shift_workflow import DimensionShiftWorkflow

logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

@dataclass
class DimensionInfo:
    """Information about each dimension"""
    id: str
    name: str
    description: str

class DimensionShiftGenerator:
    """
    Generates court case summaries with dimension shifts along 3 orthogonal dimensions.
    Creates variants for each dimension: D (Depth), P (Precision), R (Procedural)
    Each dimension has 5 levels: 0 (original) + 4 transformation levels
    """
    
    def __init__(self, gemini_api_key: str = None, llm_config: Dict[str, Any] = None):
        """Initialize with Gemini API key"""
        logger.info("Initializing DimensionShiftGenerator")
        
        api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("Gemini API key not provided")
            raise ValueError("Gemini API key not provided. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        
        logger.info("API key found, configuring LLM")
        
        # Initialize LLM configuration
        if llm_config:
            self.llm_config = llm_config.copy()
            self.llm_config["google_api_key"] = api_key
        else:
            self.llm_config = {
                "model": "gemini-2.0-flash-lite",
                "temperature": 0.1,
                "google_api_key": api_key,
                "rate_limit_rpm": 15,  # requests per minute 
            }
            
        logger.info(f"LLM config: model={self.llm_config['model']}, temp={self.llm_config['temperature']}, rate_limit={self.llm_config['rate_limit_rpm']} RPM")
        
        # Create rate limiter using LangChain's InMemoryRateLimiter
        rpm_limit = self.llm_config.get("rate_limit_rpm", 15)
        requests_per_second = rpm_limit / 60.0
        logger.info(f"Setting up rate limiter: {rpm_limit} RPM ({requests_per_second:.3f} RPS)")
        self.rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=0.1,
            max_bucket_size=rpm_limit # max tokens to be accumulated = rpm_limit
        )
        
        # Create shared rate-limited LLM instance
        logger.info("Creating rate-limited LLM instance")
        self.llm = ChatGoogleGenerativeAI(
            model=self.llm_config.get("model", "gemini-2.0-flash-lite"),
            temperature=self.llm_config.get("temperature", 0.1),
            google_api_key=api_key,
            rate_limiter=self.rate_limiter  # Apply rate limiter to LLM
        )
        
        # set up workflow without initializing it
        self.workflow = DimensionShiftWorkflow
        
        # Define dimensions (removed Structure as per README comment)
        self.dimensions = [
            DimensionInfo("Depth", "Depth ⇄ Conciseness", "Transform from detailed to concise"),
            DimensionInfo("Precision", "Technical Precision ⇄ Lay Accessibility", "Transform from technical to accessible"),
            DimensionInfo("Procedural", "Procedural Detail ⇄ Narrative Clarity", "Transform from procedural to narrative")
        ]
        
        logger.info(f"DimensionShiftGenerator initialization complete with {len(self.dimensions)} dimensions")
    
    def create_dimension_dataset(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create dimension shift dataset for a single case.
        Returns one dimension at a time as specified.
        """
        case_id = case_data.get("id", "unknown")
        summary_text = case_data.get("summary", "")
        
        logger.info(f"Starting dimension dataset creation for case {case_id}")
        
        if not summary_text:
            logger.error(f"No summary found for case {case_id}")
            raise ValueError(f"No summary found for case {case_id}")
        
        word_count = len(summary_text.split())
        logger.info(f"Case {case_id}: Processing summary with {word_count} words")
        
        print(f"Processing case {case_id}...")
        
        # Create base result structure
        result = {
            "case_id": case_id,
            "original_case_data": case_data,
            "dimension_variations": {},
            "generation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.llm_config["model"],
                "dimensions_processed": []
            }
        }
        
        # Process each dimension separately
        for dimension_info in self.dimensions:
            logger.info(f"Case {case_id}: Processing dimension {dimension_info.id} ({dimension_info.name})")
            print(f"  Processing dimension {dimension_info.id}: {dimension_info.name}")
            
            try:
                curr_workflow = self.workflow(llm=self.llm, llm_config=self.llm_config, run_name=f"{dimension_info.name} Workflow")
                logger.debug(f"Case {case_id}: Created workflow for {dimension_info.id}")
                
                # Perform multi-level shift for this dimension
                # Rate limiting is now handled automatically by LangChain
                logger.info(f"Case {case_id}: Starting multi-level shift for {dimension_info.id}")
                dimension_result = curr_workflow.multi_level_shift(
                    summary_text=summary_text,
                    dimension=dimension_info.id,
                    target_level=4,  # Go up to level 4
                    dimension_params={}  # Base parameters will be computed in workflow
                )
                
                result["dimension_variations"][dimension_info.id] = {
                    "dimension_name": dimension_info.name,
                    "dimension_description": dimension_info.description,
                    "levels": dimension_result["levels"]
                }
                
                result["generation_metadata"]["dimensions_processed"].append(dimension_info.id)
                
                # Log progress
                successful_levels = sum(1 for level_data in dimension_result["levels"].values() 
                                      if level_data.get("success", True))
                logger.info(f"Case {case_id}: Dimension {dimension_info.id} completed - {successful_levels}/5 levels successful")
                print(f"    Completed {successful_levels}/5 levels successfully")
                
            except Exception as e:
                logger.error(f"Case {case_id}: Error processing dimension {dimension_info.id}: {e}")
                print(f"    Error processing dimension {dimension_info.id}: {e}")
                result["dimension_variations"][dimension_info.id] = {
                    "dimension_name": dimension_info.name,
                    "dimension_description": dimension_info.description,
                    "error": str(e),
                    "levels": {0: {"text": summary_text, "success": True}}
                }
        
        logger.info(f"Case {case_id}: Dimension dataset creation complete")
        return result
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get current rate limiting statistics"""
        if hasattr(self, 'rate_limiter'):
            # Calculate current capacity (this is an approximation)
            return {
                "rpm_limit": self.llm_config.get("rate_limit_rpm", 15),
                "requests_per_second": self.rate_limiter.requests_per_second,
                "max_bucket_size": self.rate_limiter.max_bucket_size,
                "rate_limiter_type": "LangChain InMemoryRateLimiter"
            }
        else:
            return {"error": "Rate limiter not available"}
    
    def process_multiple_cases(self, cases_data: List[Dict[str, Any]], max_cases: int = None) -> List[Dict[str, Any]]:
        """
        Process multiple cases and return dimension shift dataset.
        """
        if max_cases:
            cases_data = cases_data[:max_cases]
            logger.info(f"Limited to {max_cases} cases (from {len(cases_data)} total)")
        
        logger.info(f"Starting batch processing of {len(cases_data)} cases")
        print(f"Processing {len(cases_data)} cases...")
        
        results = []
        successful_cases = 0
        failed_cases = 0
        
        for i, case_data in enumerate(cases_data):
            case_id = case_data.get('id', 'unknown')
            logger.info(f"Batch processing: Case {i+1}/{len(cases_data)} (ID: {case_id})")
            print(f"\n--- Case {i+1}/{len(cases_data)} ---")
            
            try:
                result = self.create_dimension_dataset(case_data)
                results.append(result)
                successful_cases += 1
                logger.info(f"Case {case_id}: Successfully processed")
                
            except Exception as e:
                logger.error(f"Case {case_id}: Processing failed - {e}")
                print(f"Failed to process case {case_id}: {e}")
                failed_cases += 1
                # Add failed case with error info
                results.append({
                    "case_id": case_id,
                    "original_case_data": case_data,
                    "error": str(e),
                    "dimension_variations": {},
                    "generation_metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "model": self.llm_config["model"],
                        "dimensions_processed": [],
                        "status": "failed"
                    }
                })
        
        logger.info(f"Batch processing complete: {successful_cases} successful, {failed_cases} failed out of {len(cases_data)} total cases")
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Total cases: {len(cases_data)}")
        print(f"Successful: {successful_cases}")
        print(f"Failed: {failed_cases}")
        
        return results
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """Save dataset to JSON file"""
        logger.info(f"Saving dataset with {len(dataset)} cases to {filename}")
        
        try:
            output_data = {
                "metadata": {
                    "generation_timestamp": datetime.now().isoformat(),
                    "generator_version": "1.0.0",
                    "model": self.llm_config["model"],
                    "total_cases": len(dataset),
                    "dimensions": [
                        {
                            "id": dim.id,
                            "name": dim.name,
                            "description": dim.description
                        }
                        for dim in self.dimensions
                    ]
                },
                "cases": dataset
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dataset successfully saved to {filename}")
            print(f"Dataset saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save dataset to {filename}: {e}")
            raise

def main():
    """
    Main function to demonstrate the dimension shift generator.
    """
    # Sample case data (similar structure to factual_omission_generator.py)
    sample_case = {
        "id": 11171,
        "name": "Grant v. Cuomo", 
        "filing_date": "1985-10-28",
        "state": "New York",
        "court": "New York state trial court",
        "case_defendants": [
            {"name": "Mario M. Cuomo"},
            {"name": "Cesar A. Perales"},
            {"name": "Edward I. Koch"}
        ],
        "causes": ["State law"],
        "relief_natures": ["Injunction / Injunctive-like Settlement"],
        "prevailing_party": "Mixed",
        "summary": "On October 28, 1985, New York City families with children at risk of removal to foster care, along with non-profit institutions dedicated to children's rights, filed a lawsuit under the federal Social Services Law, U.S. Code, and various state laws mandating the provision of social services, against State Department of Social Services and City Department of Social Services in the Supreme Court of New York County, Special Term. Represented by public and private counsel, the plaintiffs sought declaratory, injunctive, and monetary relief, claiming that the agencies failed to fulfill their duty to provide specified protective services as mandated by state and federal law."
    }
    
    # Initialize generator
    generator = DimensionShiftGenerator()
    
    # Create dimension dataset for sample case
    try:
        result = generator.create_dimension_dataset(sample_case)
        
        # Save individual case result
        generator.save_dataset([result], "sample_dimension_shift_dataset.json")
        
        print("\n=== SAMPLE RESULTS ===")
        for dim_id, dim_data in result["dimension_variations"].items():
            print(f"\nDimension {dim_id}: {dim_data['dimension_name']}")
            for level, level_data in dim_data["levels"].items():
                status = "✓" if level_data.get("success", True) else "✗"
                word_count = len(level_data.get("text", "").split()) if level_data.get("text") else 0
                print(f"  Level {level}: {status} ({word_count} words)")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
