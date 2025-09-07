#!/usr/bin/env python3
"""
Dimension Shift Agents for Legal Summary Transformation
"""

import re
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import textstat

TOLERANCE = 0.20

@dataclass
class DimensionConfig:
    dimension: str
    level: int
    target_params: Dict[str, Any]

class ExtractorAgent:
    """Extracts metadata from legal summaries"""
    
    # Procedural keyword lexicon with stem words and variants
    PROCEDURAL_KEYWORDS = [
        # Core procedural verbs and their variants
        'file', 'filed', 'filing', 'files',
        'submit', 'submitted', 'submitting', 'submits', 'submission',
        'argue', 'argued', 'arguing', 'argues', 'argument', 'arguments',
        'grant', 'granted', 'granting', 'grants',
        'deny', 'denied', 'denying', 'denies', 'denial',
        'dismiss', 'dismissed', 'dismissing', 'dismisses', 'dismissal',
        'rule', 'ruled', 'ruling', 'rules', 'rulings',
        'hold', 'held', 'holding', 'holds',
        'appeal', 'appealed', 'appealing', 'appeals',
        'affirm', 'affirmed', 'affirming', 'affirms', 'affirmation',
        'reverse', 'reversed', 'reversing', 'reverses', 'reversal',
        'remand', 'remanded', 'remanding', 'remands',
        'move', 'moved', 'moving', 'moves', 'motion', 'motions',
        'hear', 'heard', 'hearing', 'hears', 'hearings',
        'order', 'ordered', 'ordering', 'orders',
        'judge', 'judged', 'judging', 'judges', 'judgment', 'judgments',
        'try', 'tried', 'trial', 'trials',
        'jury', 'juries',
        'verdict', 'verdicts',
        'sentence', 'sentenced', 'sentencing', 'sentences',
        'petition', 'petitioned', 'petitioning', 'petitions', 'petitioner', 'petitioners',
        'complain', 'complaint', 'complaints', 'complainant', 'complainants',
        'summon', 'summons', 'summoned', 'summoning',
        'subpoena', 'subpoenaed', 'subpoenaing', 'subpoenas',
        'depose', 'deposition', 'depositions', 'deposed', 'deposing',
        'discover', 'discovery', 'discoveries', 'discovered', 'discovering',
        'evidence', 'evidenced', 'evidencing', 'evidentiary',
        'testify', 'testimony', 'testimonies', 'testified', 'testifying',
        'witness', 'witnessed', 'witnessing', 'witnesses',
        'object', 'objected', 'objecting', 'objects', 'objection', 'objections',
        'sustain', 'sustained', 'sustaining', 'sustains',
        'overrule', 'overruled', 'overruling', 'overrules',
        'proceed', 'proceeded', 'proceeding', 'proceedings', 'procedure', 'procedures',
        'docket', 'docketed', 'docketing', 'dockets',
        'calendar', 'calendared', 'calendaring', 'calendars',
        'schedule', 'scheduled', 'scheduling', 'schedules',
        'continue', 'continued', 'continuing', 'continues', 'continuance', 'continuances',
        # Legal entities and roles
        'court', 'courts', 'courthouse',
        'plaintiff', 'plaintiffs',
        'defendant', 'defendants',
        'attorney', 'attorneys', 'counsel', 'counselor', 'counselors',
        'magistrate', 'magistrates',
        'clerk', 'clerks',
        'bailiff', 'bailiffs',
        # Additional procedural terms
        'serve', 'served', 'serving', 'serves', 'service',
        'notice', 'noticed', 'noticing', 'notices',
        'brief', 'briefed', 'briefing', 'briefs',
        'plead', 'pleaded', 'pleading', 'pleads', 'plea', 'pleas',
        'cross-examine', 'cross-examined', 'cross-examining', 'cross-examination',
        'stipulate', 'stipulated', 'stipulating', 'stipulates', 'stipulation', 'stipulations',
        'settle', 'settled', 'settling', 'settles', 'settlement', 'settlements'
    ]
    
    def extract_metadata(self, summary_text: str) -> Dict[str, Any]:
        words = summary_text.split()
        word_count = len(words)
        
        return {
            "word_count": word_count,
            "citation_count": self._count_citations(summary_text),
            "procedure_keyword_count": self._count_procedural_keywords(summary_text),
            "procedure_keyword_frequency": self._count_procedural_keywords(summary_text) / max(1, word_count),
            "fkgl_score": self._calculate_fkgl(summary_text),
            "ncr_score": self._estimate_ncr(summary_text)
        }
    
    def _count_citations(self, text: str) -> int:
        """
        Counts legal citations by matching complete citation formats.
        """
        # This single, comprehensive pattern combines various Bluebook formats.
        # It looks for the core structure: Volume Reporter Page.
        # Case names and parentheticals are optional parts of the match.
        citation_pattern = re.compile(r"""
            # Optional Case Name: e.g., "Roe v. Wade, "
            (?:[A-Za-z\s.,'&_;\-]+ v\. \s [A-Za-z\s.,'&_;\-]+, \s+)?

            # Core Citation: Volume Reporter Page
            \d+                              # Volume number
            \s+
            (?:
                # Federal Reporters
                U\.S\.                        | # U.S. Reports
                S\.\s?Ct\.                   | # Supreme Court Reporter
                L\.\s?Ed\.\s?(?:2d|3d)        | # Lawyers' Edition
                F\.(?:2d|3d|4th)              | # Federal Reporter
                F\.\s?Supp\.\s?(?:2d|3d)      # Federal Supplement
                |
                # Regional Reporters
                A\.(?:2d|3d)                  | # Atlantic
                N\.E\.(?:2d|3d)               | # North Eastern
                N\.W\.(?:2d|3d)               | # North Western
                P\.(?:2d|3d)                  | # Pacific
                S\.E\.(?:2d|3d)               | # South Eastern
                S\.W\.(?:2d|3d)               | # South Western
                So\.(?:2d|3d)                 # Southern
                |
                # State-Specific Reporters (Examples)
                Cal\. \s? Rptr\. \s? (?:2d|3d|4th) | # California Reporter
                N\.Y\.S\. \s? (?:2d|3d)             # New York Supplement
                # NOTE: A truly comprehensive list of state reporters is very long.
            )
            \s+
            \d+                              # Page number

            # Optional Pinpoint Cite and Parenthetical (Court & Year)
            # e.g., ", 125" or "(9th Cir. 2025)"
            (?:
                , \s* \d+
            )?
            (?:
                \s* \( [^)]+ \)
            )?
        """, re.VERBOSE | re.IGNORECASE)

        matches = re.findall(citation_pattern, text)
        return len(matches)

    
    def _count_procedural_keywords(self, text: str) -> int:
        """
        Accurately counts whole procedural words using a single, efficient regex.
        This avoids overcounting substrings (e.g., 'file' within 'filed').
        """
        # Create a regex pattern by joining all keywords with '|' (OR).
        # The \b ensures that we match whole words only.
        pattern = r'\b(' + '|'.join(re.escape(k) for k in self.PROCEDURAL_KEYWORDS) + r')\b'
        
        # re.findall is case-sensitive, so we perform the search on the lowercased text.
        matches = re.findall(pattern, text.lower())
        
        return len(matches)
    
    
    def _calculate_fkgl(self, text: str) -> float:
        """Calculate Flesch-Kincaid Grade Level using textstat library"""
        if not text.strip():
            return 0.0
        return textstat.flesch_kincaid_grade(text)
    
    def _estimate_ncr(self, text: str) -> float:
        """Calculate New Dale-Chall Readability score using textstat library"""
        if not text.strip():
            return 6.0
        return textstat.dale_chall_readability_score(text)

class TransformerAgent:
    """Transforms summaries along specific dimensions using LLM"""
    
    def __init__(self, llm=None, llm_config: Dict[str, Any] = None):
        """Initialize with either a pre-configured LLM or llm_config for backward compatibility"""
        if llm is not None:
            # Use pre-configured LLM (preferred for rate limiting)
            self.llm = llm
        else:
            # Fallback to creating LLM from config (backward compatibility)
            api_key = llm_config.get("google_api_key") if llm_config else os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            self.llm = ChatGoogleGenerativeAI(
                model=llm_config.get("model", "gemini-2.0-flash-lite") if llm_config else "gemini-2.0-flash-lite",
                temperature=llm_config.get("temperature", 0.1) if llm_config else 0.1,
                google_api_key=api_key
            )
    
    def transform_summary(self, current_summary: str, original_summary: str, dimension_config: DimensionConfig, original_metadata: Dict[str, Any], failure_reason: Optional[str] = None) -> str:
        """Transform summary to next level with recursive refinement strategy"""
        system_prompt = """
            You are an expert legal editor. 
            Your task is to rewrite the <SOURCE_SUMMARY> to meet specific transformation goals, 
            while ensuring all key facts (like parties, core legal claims, and ultimate outcomes) remain factually consistent with the <ORIGINAL_SUMMARY>.
            You may draw core facts from the <ORIGINAL_SUMMARY> as needed, but you CANNOT add depth, technical or procedural details to the transformed summary.   
            Output ONLY the transformed summary body.
        """.strip()
        user_prompt = self._build_prompt(current_summary, original_summary, dimension_config, original_metadata, failure_reason)
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = self.llm.invoke(messages)
        return response.content.strip()
    
    def _build_prompt(self, current_summary: str, original_summary: str, config: DimensionConfig, original_metadata: Dict[str, Any], failure_reason: Optional[str] = None) -> str:
        """Build dimension-specific prompt with recursive refinement strategy"""
        # Add retry context if available
        retry_note = f"<RETRY_CONTEXT>Previous validation failed due to: {failure_reason}. Pay close attention to this feedback and address the issue in your new response.</RETRY_CONTEXT>" if failure_reason else ""
        base_instruction = retry_note + f"""
            <SOURCE_SUMMARY>
            {current_summary}
            </SOURCE_SUMMARY>

            <ORIGINAL_SUMMARY>
            {original_summary}
            </ORIGINAL_SUMMARY>
        """.strip()
        
        if config.dimension == "Depth":  # Depth
            return self._depth_prompt(base_instruction, config, original_metadata)
        elif config.dimension == "Precision":  # Precision
            return self._precision_prompt(base_instruction, config, original_metadata)
        elif config.dimension == "Procedural":  # Procedural
            return self._procedural_prompt(base_instruction, config, original_metadata)
        else:
            raise ValueError(f"Unknown dimension: {config.dimension}")
    
    def _depth_prompt(self, base_instruction: str, config: DimensionConfig, original_metadata: Dict[str, Any]) -> str:
        target_pct = config.target_params.get("word_target_percent", 85)
        target_word_count = int(original_metadata["word_count"] * target_pct / 100)
        level_details = {
            1: {
                "persona": "Senior Associate Attorney",
                "rule": "Condense without losing substance. 1) Combine sentences describing a single event. 2) Remove dates of non-dispositive procedural filings (e.g., keep the complaint date, remove motion-filing dates). 3) Replace lists of individual officials (e.g., 'Jesse H. Cutrer, Arnold Spiers, Marshall Holloway...') with their collective title ('members of the Commission Council')."
            },
            2: {
                "persona": "Managing Editor of a Legal Journal",
                "rule": "Summarize procedural clusters. Replace sequences of motions and responses (e.g., 'motion to dismiss, opposition, reply, hearing') with a single phrase describing the outcome (e.g., 'After the court denied the defendants' motion to dismiss...'). Focus only on the complaint, dispositive motions, and final judgment."
            },
            3: {
                "persona": "In-House Counsel briefing an executive",
                "rule": "Reformat for scannability. Rewrite as an 'Executive Summary': A single sentence for the overall outcome, followed by 3-4 bullet points detailing the key results (e.g., who won, what relief was granted, any monetary award)."
            },
            4: {
                "persona": "News Headline Writer",
                "rule": "Distill to the absolute essence. Create a single, impactful headline (max 15 words). Below it, add exactly three bullet points: **Who Sued Who?**, **Why?**, **What Happened?**."
            }
        }
        
        negative_constraint = "Your primary goal is reducing length. **Preserve** the original level of technical vocabulary and procedural focus as much as the word count allows. Do not simplify legal terms or shift the narrative focus."
        
        details = level_details.get(config.level, {"persona": "Legal Editor", "rule": "Reduce length while maintaining legal precision."})
        
        return f"""CONCISENESS TRANSFORMATION:
            Target: Reduce the summary to approximately {target_word_count} words.
            Persona: Imagine you are a {details["persona"]}.
            Scenario: {details["rule"]}
            Constraint: {negative_constraint}

            Transform the <SOURCE_SUMMARY> based on these instructions.

            {base_instruction}"""
    
    def _precision_prompt(self, base_instruction: str, config: DimensionConfig, original_metadata: Dict[str, Any]) -> str:
        grade_target = config.target_params.get("grade_target", (11, 12))
        citation_target = config.target_params.get("citation_target_percent", 100)
        citation_target_count = int(original_metadata["citation_count"] * citation_target / 100)

        level_details = {
            1: {
                "persona": "Senior Lawyer editing a junior's draft for a client memo",
                "rule": "Improve clarity for a legally sophisticated but non-specialist audience. Simplify sentence structure and replace dense jargon with common professional equivalents. **Example:** change *'The County sought to be exempted from the preclearance procedures of Section 5 of the Voting Rights Act'* to *'The County asked for an exemption from the Voting Rights Act's 'preclearance' rule, which required federal approval for any new voting laws.'*",
                "grade": "12"
            },
            2: {
                "persona": "Journalist for the Associated Press",
                "rule": "Explain legal concepts simply. Remove statute citations and replace them with the law's common name. Explain key legal terms in plain English. **Example:** change *'sought declaratory relief against the Attorney General pursuant to Section 4(a) of the Act'* to *'sued the Attorney General, asking the court to officially declare them exempt under a provision of the Voting Rights Act known as the 'bailout' clause.'*",
                "grade": "10"
            },
            3: {
                "persona": "High School Civics Teacher",
                "rule": "Make it relatable. Assume the reader has no legal background. Use simple analogies where helpful. **Example:** change *'The consent judgment and decree exempted Merced County from the preclearance requirements'* to *'The court approved an agreement that meant Merced County no longer had to ask the federal government for permission before it could change its voting rules.'*",
                "grade": "9"
            },
            4: {
                "persona": "Writer for a 'News for Kids' website",
                "rule": "Tell it like a simple story. Use very short sentences and basic vocabulary. Avoid any term that would need a dictionary. **Example:** change *'The plaintiffs alleged that, in deliberately failing to adequately screen those entering the DCJ...'* to *'A man sued the jail. He said the jail didn't check new inmates for a skin problem called scabies. Because of this, the problem spread to everyone.'*",
                "grade": "8"
            }
        }

        details = level_details.get(config.level, {"persona": "Legal Writer", "rule": "Use standard legal vocabulary.", "grade": "13+"})
        rule, grade_display = details["rule"], details["grade"]

        negative_constraint = "Your primary goal is making the language more accessible. **Preserve** the original length and level of procedural detail as much as possible. Do not cut significant facts; rewrite them in simpler terms."
        
        # Format grade target for display
        if isinstance(grade_target, tuple):
            grade_display = f"{grade_target[0]}-{grade_target[1]}"
        else:
            grade_display = str(grade_target)
        
        return f"""ACCESSIBILITY TRANSFORMATION:
            Target: Rewrite as if the readers are Grade {grade_display}. Slightly reduce the usage of difficult words, and shorten sentence lengths.
            Persona: Imagine you are a {details["persona"]}.
            Scenario: {details["rule"]}
            Constraint: {negative_constraint}

            Transform the SOURCE summary based on these instructions.

            {base_instruction}"""
    
    def _procedural_prompt(self, base_instruction: str, config: DimensionConfig, original_metadata: Dict[str, Any]) -> str:
        keyword_target = config.target_params.get("keyword_target_percent", 75)
        keyword_target_count = int(original_metadata["procedure_keyword_count"] * keyword_target / 100)
        
        level_details = {
            1: {
                "persona": "Legal Historian",
                "rule": "Explain the 'why' behind the 'what'. For each major procedural step (complaint, major motion, judgment), explain its strategic purpose. **Guiding Questions:** Why did the plaintiffs file this case? What were the defendants trying to achieve with their motion to dismiss?",
                
            },
            2: {
                "persona": "Law Student writing a case brief",
                "rule": "Center the legal question and the court's answer. Start by stating the central legal conflict. Summarize the procedural history in one sentence (e.g., 'After filing the initial complaint, the parties filed cross-motions...'). Dedicate the rest of the summary to the court's reasoning and the final holding.",
                
            },
            3: {
                "persona": "Policy Analyst for an advocacy group",
                "rule": "Focus on the outcome and its real-world impact. Start with the final result. Then, explain what that decision meant for the parties involved and for the law or policy being challenged. The court process should only be mentioned in passing. **Example:** Focus on *'The jail must now change its policies to ensure inmates with disabilities can access programs and services.'* rather than *'The court approved a settlement agreement following a motion for preliminary approval.'*",

            },
            4: {
                "persona": "Feature Writer for a magazine",
                "rule": "Tell the story of the people. Completely remove legal and procedural terms. Use a three-act structure: **1. The Problem:** Describe the human situation that started the conflict (e.g., 'Prisoners with wheelchairs were stuck in their cells...'). **2. The Fight:** Briefly state that they turned to the legal system for help. **3. The Resolution:** Describe how their situation changed because of the case's outcome.",
            }
        }
        
        negative_constraint = "Focus only on shifting the focus from procedural details to narrative clarity. Do not significantly change the word count or vocabulary complexity unless necessary for the narrative shift."
        
        details = level_details.get(config.level, {"persona": "Legal Analyst", "rule": "Reduce emphasis on procedures."})
        
        return f"""NARRATIVE TRANSFORMATION: 
            Target: Reduce the emphasis on legal procedures and court actions, and increase the focus on the narrative story of the case.
            Persona: Imagine you are a {details["persona"]}.
            Scenario: {details["rule"]}
            Constraint: {negative_constraint}

            Transform the SOURCE summary based on these instructions.

            {base_instruction}"""

class ValidatorAgent:
    """Validates transformations with hard checks and LLM pairwise critique"""
    
    def __init__(self, llm=None, llm_config: Dict[str, Any] = None):
        """Initialize with either a pre-configured LLM or llm_config for backward compatibility"""
        if llm is not None:
            # Use pre-configured LLM (preferred for rate limiting)
            self.llm = llm
        else:
            # Fallback to creating LLM from config (backward compatibility)
            api_key = llm_config.get("google_api_key") if llm_config else os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            self.llm = ChatGoogleGenerativeAI(
                model=llm_config.get("model", "gemini-2.0-flash-lite") if llm_config else "gemini-2.0-flash-lite",
                temperature=llm_config.get("temperature", 0.1) if llm_config else 0.1,
                google_api_key=api_key
            )
        self.extractor = ExtractorAgent()
    
    def validate_transformation(self, transformed_text: str, dimension_config: DimensionConfig, 
                              original_metadata: Dict[str, Any], original_text: str, 
                              previous_text: str) -> Dict[str, Any]:
        """Validate transformation with hard checks and LLM critique"""
        
        # Extract new metadata
        new_metadata = self.extractor.extract_metadata(transformed_text)
        
        # Hard checks
        hard_check_result = self._hard_checks(new_metadata, original_metadata, dimension_config)
        
        # LLM pairwise critique
        llm_critique_result = self._llm_pairwise_critique(
            transformed_text, previous_text, original_text, dimension_config
        )
        
        # Overall validation result
        passed = hard_check_result["passed"] and llm_critique_result["passed"]
        
        # Create a unified failure reason string for transformer feedback
        failure_reason_str = None
        if not passed:
            reasons = []
            if not hard_check_result["passed"]:
                hard_reason = hard_check_result.get("failure_reason")
                if isinstance(hard_reason, dict):
                    # Extract non-None values from dict
                    hard_reasons = [v for v in hard_reason.values() if v is not None]
                    if hard_reasons:
                        reasons.extend(hard_reasons)
                elif hard_reason:
                    reasons.append(str(hard_reason))
            
            if not llm_critique_result["passed"]:
                llm_reason = llm_critique_result.get("failure_reason")
                if llm_reason:
                    reasons.append(str(llm_reason))
            
            failure_reason_str = "; ".join(reasons) if reasons else "Validation failed"
        
        return {
            "passed": passed,
            "new_metadata": new_metadata,
            "hard_checks": hard_check_result,
            "llm_critique": llm_critique_result,
            "failure_reason": failure_reason_str  # Unified string for transformer
        }
    
    def _hard_checks(self, new_metadata: Dict[str, Any], original_metadata: Dict[str, Any], 
                    config: DimensionConfig) -> Dict[str, Any]:
        """Perform dimension-specific hard checks"""
        
        if config.dimension == "Depth":  # Depth checks
            target_pct = config.target_params.get("word_target_percent", 85)
            target_words = int(original_metadata["word_count"] * target_pct / 100)
            tolerance = int(target_words * TOLERANCE)  # ±k% tolerance
            
            actual_words = new_metadata["word_count"]
            passed = abs(actual_words - target_words) <= tolerance
            
            return {
                "passed": passed,
                "target_words": target_words,
                "actual_words": actual_words,
                "tolerance": tolerance,
                "failure_reason": f"Word count {actual_words} not within ±{TOLERANCE*100}% of target {target_words}" if not passed else None
            }
            
        elif config.dimension == "Precision":  # Precision checks
            citation_target_pct = config.target_params.get("citation_target_percent", 100)
            level = config.target_params.get("level", 1)
            previous_metadata = config.target_params.get("previous_metadata", {})
            
            # Citation count check with 20% tolerance
            # target_citations = int(original_metadata["citation_count"] * citation_target_pct / 100)
            # citation_tolerance = int(target_citations * TOLERANCE)  # 20% tolerance
            # citation_passed = new_metadata["citation_count"] <= (target_citations + citation_tolerance)
            citation_passed = new_metadata["citation_count"] <= previous_metadata["citation_count"]
            
            if level >= 1 and (previous_metadata or level == 1):
                # Relative thresholds: FKGL and NCR must be lower than previous level
                # For level 1, compare against original (level 0) metadata
                if level == 1:
                    prev_fkgl = original_metadata.get("fkgl_score", float('inf'))
                    prev_ncr = original_metadata.get("ncr_score", float('inf'))
                else:
                    prev_fkgl = previous_metadata.get("fkgl_score", float('inf'))
                    prev_ncr = previous_metadata.get("ncr_score", float('inf'))
                
                fkgl_passed = new_metadata["fkgl_score"] < prev_fkgl
                ncr_passed = new_metadata["ncr_score"] < prev_ncr
                
                failure_reasons = []
                if not fkgl_passed:
                    failure_reasons.append(f"FKGL score {new_metadata['fkgl_score']:.1f} not lower than previous level {prev_fkgl:.1f}")
                if not ncr_passed:
                    failure_reasons.append(f"NCR score {new_metadata['ncr_score']:.1f} not lower than previous level {prev_ncr:.1f}")
                if not citation_passed:
                    # failure_reasons.append(f"Citation count {new_metadata['citation_count']} exceeds target {target_citations} (±{citation_tolerance} tolerance)")
                    failure_reasons.append(f"Citation count {new_metadata['citation_count']} more than previous level {previous_metadata['citation_count']}")
            else:
                # Fallback (should not happen with new logic)
                fkgl_passed = True
                ncr_passed = True
                citation_passed = True
                failure_reasons = []
                # if not citation_passed:
                #     failure_reasons.append(f"Citation count {new_metadata['citation_count']} exceeds target {target_citations} (±{citation_tolerance} tolerance)")

            # Overall pass condition (Either FKGL Readability or NCR score, and citation count)
            passed = (fkgl_passed or ncr_passed) and citation_passed 

            return {
                "passed": passed,
                "fkgl_passed": fkgl_passed,
                "ncr_passed": ncr_passed,
                "citation_passed": citation_passed,
                # "target_citations": target_citations,
                # "citation_tolerance": citation_tolerance,
                "failure_reason": "; ".join(failure_reasons) if failure_reasons else None
            }
            
        elif config.dimension == "Procedural":  # Procedural checks
            level = config.target_params.get("level", 1)
            previous_metadata = config.target_params.get("previous_metadata", {})
            
            if level >= 1:
                # Relative thresholds: both procedural keyword frequency and term count must be lower than previous level
                # For level 1, compare against original (level 0) metadata
                if level == 1:
                    prev_frequency = original_metadata.get("procedure_keyword_frequency", float('inf'))
                    prev_term_count = original_metadata.get("procedure_keyword_count", float('inf'))
                else:
                    prev_frequency = previous_metadata.get("procedure_keyword_frequency", float('inf'))
                    prev_term_count = previous_metadata.get("procedure_keyword_count", float('inf'))
                
                current_frequency = new_metadata["procedure_keyword_frequency"]
                current_term_count = new_metadata["procedure_keyword_count"]
                
                # Either frequency OR term count criteria must be met for the level to pass
                frequency_passed = current_frequency < prev_frequency
                term_count_passed = current_term_count < prev_term_count
                passed = frequency_passed or term_count_passed
                
                # Build failure reason if both checks failed
                failure_reasons = []
                if not frequency_passed:
                    failure_reasons.append(f"Procedural keyword frequency {current_frequency:.4f} not lower than previous level {prev_frequency:.4f}")
                if not term_count_passed:
                    failure_reasons.append(f"Procedural term count {current_term_count} not lower than previous level {prev_term_count}")
                
                failure_reason = "; ".join(failure_reasons) if not passed else None
                
                return {
                    "passed": passed,
                    "frequency_passed": frequency_passed,
                    "term_count_passed": term_count_passed,
                    "current_frequency": current_frequency,
                    "previous_frequency": prev_frequency,
                    "current_term_count": current_term_count,
                    "previous_term_count": prev_term_count,
                    "failure_reason": failure_reason
                }
            else:
                # Fallback (should not happen with new logic)
                return {"passed": True}
        
        return {"passed": True}
    
    def _llm_pairwise_critique(self, transformed_text: str, previous_text: str, 
                              original_text: str, config: DimensionConfig) -> Dict[str, Any]:
        """Perform LLM pairwise critique for factual consistency and dimension-specific checks"""
        
        dimension_prompts = {
            "Depth": "Does <NEW_SUMMARY> omit low-importance details while retaining the core factual basis and outcome of the case vs <SOURCE_SUMMARY>? YES/NO. If NO, identify any high-importance facts that were incorrectly dropped.",
            "Precision": "Does <NEW_SUMMARY> use more accessible & simplified language than <SOURCE_SUMMARY> without being misleading? YES/NO. If NO, identify any legal jargon or complex terms that were not simplified.",
            "Procedural": "Does the <NEW_SUMMARY> focus more on the parties involved, the real-world context, and the ultimate outcome's impact, rather than on court actions and legal reasoning vs <SOURCE_SUMMARY>? YES/NO. If No, provide one-sentence explanation."
        }
        
        system_prompt = """You are a legal expert evaluating summary transformations. Use chain of thought reasoning and provide clear YES/NO answers."""
        
        user_prompt = f"""You will compare three summaries of a legal case: a <SOURCE_SUMMARY>, <NEW_SUMMARY>, and <ORIGINAL_SUMMARY>.

<ORIGINAL_SUMMARY>
{original_text}
</ORIGINAL_SUMMARY>

<SOURCE_SUMMARY>
{previous_text}
</SOURCE_SUMMARY>
 
<NEW_SUMMARY>
{transformed_text}
</NEW_SUMMARY>

Evaluate using this chain of thought:

1. First, identify the core facts present in the <ORIGINAL_SUMMARY> (parties, core dispute, outcome).

2. Second, check if these core facts are all present in the <NEW_SUMMARY>.

3. Third, dimension-specific check: {dimension_prompts.get(config.dimension, "Perform general quality check.")}

Format your response as: 
Factual Consistency: YES/NO - [brief explanation if No]
Dimension Check: YES/NO - [brief explanation if No] 

Do NOT include thinking steps, and do not apply any additional formatting to your response.
"""
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = self.llm.invoke(messages)
        
        # Parse response
        response_text = response.content.strip()
        
        # Extract all YES/NO responses and failure reasons using regex
        # More flexible pattern that handles various formats
        yes_no_pattern = r'(Factual Consistency|Dimension Check):\s*(YES|NO)\s*(?:-\s*)?(.*?)(?=\n(?:Factual Consistency|Dimension Check)|$)'
        matches = re.findall(yes_no_pattern, response_text, re.IGNORECASE | re.DOTALL)

        factual_consistency = None
        dimension_check = None
        

        for label, verdict, explanation in matches:
            if "factual consistency" in label.lower():
                factual_consistency = {"verdict": verdict.upper(), "explanation": explanation.strip() if explanation else ""}
            elif "dimension check" in label.lower():
                dimension_check = {"verdict": verdict.upper(), "explanation": explanation.strip() if explanation else ""} 
        
        # Fallback parsing if regex didn't match expected format
        if factual_consistency is None or dimension_check is None:
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Try simpler patterns
                if 'factual consistency' in line.lower():
                    if 'yes' in line.lower():
                        factual_consistency = {"verdict": "YES", "explanation": ""}
                    elif 'no' in line.lower():
                        # Extract explanation after NO
                        parts = line.lower().split('no', 1)
                        explanation = parts[1].strip(' -') if len(parts) > 1 else ""
                        factual_consistency = {"verdict": "NO", "explanation": explanation}
                        
                elif 'dimension check' in line.lower():
                    if 'yes' in line.lower():
                        dimension_check = {"verdict": "YES", "explanation": ""}
                    elif 'no' in line.lower():
                        # Extract explanation after NO
                        parts = line.lower().split('no', 1)
                        explanation = parts[1].strip(' -') if len(parts) > 1 else ""
                        dimension_check = {"verdict": "NO", "explanation": explanation} 
        
        # Determine overall pass status   
        passed = (factual_consistency and factual_consistency["verdict"] == "YES" and 
                    dimension_check and dimension_check["verdict"] == "YES")
        
        final_verdict = "PASS" if passed else "FAIL"

        # Get failure reason
        failure_reason = ""
        if not passed:
            # Check if we failed to parse the response entirely
            if factual_consistency is None and dimension_check is None:
                failure_reason = f"Unable to parse validation response. Raw response: {response_text}"
            else:
                # Build failure reasons for each failed check
                failure_parts = []
                if factual_consistency and factual_consistency["verdict"] == "NO":
                    failure_parts.append("Core Fact Missing: " + factual_consistency.get("explanation", "No explanation provided"))
                if dimension_check and dimension_check["verdict"] == "NO":
                    failure_parts.append("Dimension Not Shifted: " + dimension_check.get("explanation", "No explanation provided"))
                
                # If no specific failures but still not passed, it might be a parsing issue
                if not failure_parts:
                    failure_parts.append(f"Validation failed but no specific reasons found. Response: {response_text}")
                
                failure_reason = "; ".join(failure_parts)
        
        return {
            "passed": passed,
            "critique_text": response_text, 
            "factual_consistency": factual_consistency,
            "dimension_check": dimension_check,
            "final_verdict": final_verdict,
            "failure_reason": failure_reason
        }
    