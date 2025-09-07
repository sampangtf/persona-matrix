
import sys
import os
import json
from textstat import flesch_kincaid_grade, dale_chall_readability_score
from bs4 import BeautifulSoup
import glob
import re
import argparse
from datetime import datetime
import statistics

from dotenv import load_dotenv
from pprint import pprint
load_dotenv()

def parse_args(): 
    version_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description='Run dimension shift dataset generation')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file path')
    parser.add_argument('--output', '-o', default=f'../transformed_dataset/shifted_summaries_v1_{version_tag}.json', help='Output JSON file path')
    parser.add_argument('--max-cases', '-m', type=int, help='Maximum number of cases to process')
    parser.add_argument('--processed-file', '-p', default='processed_cases.txt', help='File to track processed case IDs')
    return parser.parse_args()

def load_processed_cases(processed_file):
    """Load list of already processed case IDs"""
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            processed = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(processed)} processed case IDs from {processed_file}")
        return processed
    else:
        print(f"No processed file found at {processed_file}, starting fresh")
        return set()

def save_processed_case(processed_file, case_id):
    """Append a processed case ID to the tracking file"""
    with open(processed_file, 'a') as f:
        f.write(f"{case_id}\n")


def count_words(text: str) -> int:
    """
    Count the number of words in the given text, ignoring punctuation and HTML tags.
    
    Steps:
    1. Remove HTML tags using a regex that targets anything between < and >.
    2. Remove punctuation by keeping only word characters and whitespace.
    3. Split the cleaned text on whitespace to get individual words.
    4. Return the count of words.
    """
    # Remove HTML tags
    no_tags = re.sub(r'<[^>]+>', '', text)
    # Remove punctuation
    no_punct = re.sub(r'[^\w\s]', '', no_tags)
    # Split into words and count
    words = no_punct.split()
    return len(words)


def compute_summary_statistics(cases):
    """Compute and print summary statistics on readability and length"""
    print("\n=== SUMMARY STATISTICS ===")
    
    summaries = [case.get('summary', '') for case in cases if case.get('summary')]
    if not summaries:
        print("No summaries found!")
        return
    
    # Length statistics
    word_counts = [count_words(summary) for summary in summaries]
    print(f"Word Count Statistics (n={len(word_counts)}):")
    print(f"  Mean: {statistics.mean(word_counts):.1f}")
    print(f"  Median: {statistics.median(word_counts):.1f}")
    print(f"  Min: {min(word_counts)}")
    print(f"  Max: {max(word_counts)}")
    if len(word_counts) > 1:
        print(f"  Std Dev: {statistics.stdev(word_counts):.1f}")
    else:
        print("  Std Dev: N/A (only one case)")
    
    # Readability statistics
    print("\nReadability Statistics:")
    fk_grades = []
    dc_scores = []
    
    for i, summary in enumerate(summaries[:100]):  # Sample first 100 for speed
        try:
            fk_grade = flesch_kincaid_grade(summary)
            dc_score = dale_chall_readability_score(summary)
            fk_grades.append(fk_grade)
            dc_scores.append(dc_score)
        except:
            continue
    
    if fk_grades:
        print(f"Flesch-Kincaid Grade Level (n={len(fk_grades)}):")
        print(f"  Mean: {statistics.mean(fk_grades):.1f}")
        print(f"  Median: {statistics.median(fk_grades):.1f}")
        print(f"  Range: {min(fk_grades):.1f} - {max(fk_grades):.1f}")
    
    if dc_scores:
        print(f"Dale-Chall Readability Score (n={len(dc_scores)}):")
        print(f"  Mean: {statistics.mean(dc_scores):.1f}")
        print(f"  Median: {statistics.median(dc_scores):.1f}")
        print(f"  Range: {min(dc_scores):.1f} - {max(dc_scores):.1f}")
    
    print("=" * 40)


def main():
    args = parse_args()
    
    print("current working directory:", os.getcwd())
    # add file parent directory to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Added parent directory to sys.path.: ", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from transform.dimension_shift_generator import DimensionShiftGenerator
    print("Imported DimensionShiftGenerator.")

    # Load processed cases for resume functionality
    processed_cases = load_processed_cases(args.processed_file)
    
    # Load cases from input JSON file
    print(f"Loading cases from {args.input}")
    with open(args.input, 'r') as f:
        all_cases = json.load(f)
    print(f"Total cases loaded: {len(all_cases)}")
    
    # Filter out already processed cases
    unprocessed_cases = [case for case in all_cases if str(case.get('id', '')) not in processed_cases]
    print(f"Unprocessed cases: {len(unprocessed_cases)} (skipping {len(all_cases) - len(unprocessed_cases)} already processed)")
    
    # Apply max cases limit if specified
    if args.max_cases and args.max_cases < len(unprocessed_cases):
        unprocessed_cases = unprocessed_cases[:args.max_cases]
        print(f"Limited to first {args.max_cases} unprocessed cases")
    
    if not unprocessed_cases:
        print("No cases to process!")
        return
    
    # Compute summary statistics
    compute_summary_statistics(unprocessed_cases)
    
    # Initialize dimension shift generator
    model_name = "gemini-2.5-flash"
    
    llm_config = {
        "model": model_name,
        "temperature": 0.25,
        "rate_limit_rpm": 10,  # requests per minute
        "google_api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    }
    
    print(f"\nInitializing DimensionShiftGenerator with model: {model_name}")
    generator = DimensionShiftGenerator(llm_config=llm_config)
    
    # Process cases
    print(f"\n=== STARTING MAIN PROCESSING ===")
    print(f"Processing {len(unprocessed_cases)} cases...")
    
    results = []
    successful_cases = 0
    failed_cases = 0
    
    for i, case in enumerate(unprocessed_cases):
        case_id = case.get('id', 'unknown')
        print(f"\n--- Case {i+1}/{len(unprocessed_cases)} (ID: {case_id}) ---")
        
        try:
            # Process single case
            result = generator.create_dimension_dataset(case)
            
            # Validate that all dimensions and levels have valid summaries
            case_valid = True
            failure_reasons = []
            
            # Quick check of results and validation
            print("=" * 15,"Quick Check:", "=" * 15)
            for dim_id, dim_data in result["dimension_variations"].items():
                print(f"\nDimension {dim_id}: {dim_data.get('dimension_name', dim_id)}")
                for level, level_data in dim_data["levels"].items():
                    success = level_data.get("success", True)
                    if "text" in level_data:
                        text = level_data["text"]
                    else:
                        text = level_data.get("transformed_text", "")
                    word_count = count_words(text)
                    
                    # Check for failures
                    if not success:
                        case_valid = False
                        failure_reasons.append(f"Dim {dim_id} Level {level}: unsuccessful")
                    elif word_count == 0:
                        case_valid = False
                        failure_reasons.append(f"Dim {dim_id} Level {level}: zero words")
                    elif not text or text.strip() == "":
                        case_valid = False
                        failure_reasons.append(f"Dim {dim_id} Level {level}: empty text")
                    
                    status = "✓" if success and word_count > 0 else "✗"
                    print(f"  Level {level}: {status} ({word_count} words)")
            print("=" * 40)
            
            if case_valid:
                # Add to results and mark as processed if completely valid
                results.append(result)
                save_processed_case(args.processed_file, case_id)
                successful_cases += 1
                print(f"✓ Case {case_id} completed successfully")
            else:
                # Save to results but don't mark as processed - will retry on next run
                results.append(result)  # Save failed results for analysis
                failed_cases += 1
                print(f"✗ Case {case_id} failed validation:")
                for reason in failure_reasons:
                    print(f"    - {reason}")
                print("  Saved to results but not marked as processed - will retry on next run")
            
        except Exception as e:
            print(f"✗ Case {case_id} failed: {e}")
            failed_cases += 1
            continue
    
    # Save results
    if results:
        print(f"\nSaving {len(results)} results to {args.output}")
        generator.save_dataset(results, args.output)
    
    # Final summary
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total cases processed: {len(unprocessed_cases)}")
    print(f"Successful: {successful_cases}")
    print(f"Failed: {failed_cases}")
    print(f"Results saved to: {args.output}")
    print(f"Processed cases tracked in: {args.processed_file}")

if __name__ == "__main__":
    main()