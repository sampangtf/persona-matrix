"""
Persona definitions for legal document evaluation.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass 
from src.models.state import Criterion, Task
from typing import Any

# @dataclass
# class PersonaEvaluationResult:
#     """Results from evaluating a single persona."""
#     persona_name: str
#     persona_description: str
#     criteria: List[Criterion]
#     quantification_results: Dict[str, Any]
#     final_score: Optional[float]
#     errors: List[str]
#     raw_responses: Dict[str, str]


# @dataclass
# class PersonaEvalResult:
#     """Complete results from multi-persona evaluation."""
#     workflow_id: str
#     task: Task
#     test_case: str
#     ground_truth: Optional[str]
#     persona_results: Dict[str, PersonaEvaluationResult]
#     summary_report: str
#     overall_statistics: Dict[str, Any]
#     errors: List[str]

@dataclass
class Persona:
    """Represents a user persona for evaluation."""
    name: str
    description: str
    representative_users: List[str]
    informational_needs: List[str]
    key_evaluation_dimensions: List[str]
    mission_alignment: str
    usage: str
    
#     def get_context_prompt(self) -> str:
#         """Generate context prompt for this persona."""
#         context = f"""
# You are evaluating from the perspective of: {self.name}

# Representative Users: {', '.join(self.representative_users)}

# Distinct Informational Needs:
# {chr(10).join(f'• {need}' for need in self.informational_needs)}

# Key Evaluation Dimensions:
# {chr(10).join(f'• {dimension}' for dimension in self.key_evaluation_dimensions)}

# Mission Alignment: {self.mission_alignment}
# """
    def get_context_prompt(self) -> str:
        """Generate context prompt for this persona."""
        context = f"""
You are generating evaluation criteria from the perspective of: {self.name}
Based on how they utilize the legal summary: {self.usage}
""".strip()
        return context


# Define the 6 legal document personas
RAW_LEGAL_DOCUMENT_PERSONAS = { 
        "litigation_professional": {
            "name": "Litigation Professional",
            "description": "Civil rights litigators who need precise legal information for case preparation.",
            "representative_users": [
                "Civil rights litigators",
                "Defense attorneys",
                "Plaintiff attorneys"
            ],
            "informational_needs": [
                "Precedent value and authority",
                "Procedural posture clarity",
                "Relief granted details",
                "Precise legal citations",
                "Factual allegations",
                "Court holdings and reasoning"
            ],
            "key_evaluation_dimensions": [
                "Legal citation accuracy",
                "Completeness of procedural details",
                "Procedural clarity",
                "Precedential value identification",
                "Relief and remedies precision"
            ],
            "mission_alignment": "Civil rights lawyers can use Clearinghouse to save billable research time and locate hard-to-obtain filings, especially benefiting those without commercial database access.",
            "usage": "Litigation professionals will refer to the legal summary to find a case's holdings and procedural history for use in their own motions and briefs."
        },
        "legal_education": {
            "name": "Legal Education Community",
            "description": "Law professors, teachers, and students who need educational content.",
            "representative_users": [
                "Law professors",
                "High school teachers",
                "Law students",
                "Legal educators"
            ],
            "informational_needs": [
                "Issue-rule-application breakdown (IRAC)",
                "Doctrinal trends and developments",
                "Illustrative fact patterns",
                "Teaching-friendly case summaries",
                "Legal principle explanations",
                "Comparative case analysis"
            ],
            "key_evaluation_dimensions": [
                "Concise IRAC structure",
                "Illustrative excerpts quality",
                "Educational value",
                "Clarity for learning",
                "Doctrinal significance"
            ],
            "mission_alignment": "Clearinghouse offers specific sessions for teachers and students, making them key target users for educational content.",
            "usage": "The legal education community will use the summary in teaching materials to present the application of legal principles and as a reading assignment for students to learn the core parts of a case."
        },
        "journalism_media": {
            "name": "Journalism & Media",
            "description": "Court reporters, legal affairs editors, and fact-checkers who need accessible information.",
            "representative_users": [
                "Court reporters",
                "Legal affairs editors",
                "Fact-checkers",
                "News writers",
                "Broadcast journalists"
            ],
            "informational_needs": [
                "Headline facts and key takeaways",
                "Plain-language legal significance",
                "Quick verification links and sources",
                "Newsworthy angles",
                "Public interest elements",
                "Timeline of events"
            ],
            "key_evaluation_dimensions": [
                "Clarity and accessibility",
                "Brevity and conciseness",
                "Newsworthiness identification",
                "Accessible language usage",
                "Fact verification support"
            ],
            "mission_alignment": "Free public documents make stories verifiable and align with Clearinghouse's mission to guarantee free access to information for public dissemination.",
            "usage": "Journalism and media professionals will pull key facts and the outcome of a court decision from the legal summary for use in their reporting."
        },
        "policy_advocacy": {
            "name": "Policy & Advocacy Stakeholders",
            "description": "Civil rights NGOs, think tanks, and legislators focused on policy implications.",
            "representative_users": [
                "Civil rights NGOs",
                "Think tank researchers",
                "Legislators and staff",
                "Policy advocates",
                "Campaign organizations"
            ],
            "informational_needs": [
                "Trend data and patterns",
                "Comparable cases and precedents",
                "Statutory context and implications",
                "Enforceability details",
                "Policy impact assessment",
                "Jurisdictional variations"
            ],
            "key_evaluation_dimensions": [
                "Impact framing accuracy",
                "Jurisdictional trends identification",
                "Policy implications clarity",
                "Comparative case context",
                "Enforcement practicality"
            ],
            "mission_alignment": "Originally created for advocates and policymakers, Clearinghouse serves NGOs tracking litigation to steer campaigns and policy development.",
            "usage": "Policy and advocacy stakeholders will review the legal summary to find out about the results of a court case and use this information to help shape their advocacy work and policy ideas."
        },
        "public_self_help": {
            "name": "Public Self-Help Users",
            "description": "Pro se litigants and individuals representing themselves without attorneys.",
            "representative_users": [
                "Pro se litigants",
                "Self-represented parties",
                "Community advocates",
                "Individuals seeking legal guidance"
            ],
            "informational_needs": [
                "Plain-language summary of legal concepts",
                "Procedural steps and requirements",
                "Filing deadlines and timelines",
                "Pro bono contact information",
                "Practical next steps",
                "Rights and obligations explanation"
            ],
            "key_evaluation_dimensions": [
                "Readability and comprehension",
                "Actionable next steps",
                "Avoidance of legal jargon",
                "Practical guidance provision",
                "Resource accessibility"
            ],
            "mission_alignment": "32% of federal civil rights case plaintiffs are pro se (1998-2017). They represent concerned communities whose civil rights may be infringed and need Clearinghouse's free information most.",
            "usage": "Public self-help users will consult the legal summary to learn about the issues and outcomes in cases that might relate to their own legal matters and to look for information that could help them decide what to do next."
        },
        "academic_research": {
            "name": "Academic & Data-Science Researchers",
            "description": "Empirical legal scholars and NLP researchers focused on data analysis.",
            "representative_users": [
                "Empirical legal scholars",
                "NLP researchers",
                "Data scientists",
                "Academic researchers",
                "Social science researchers"
            ],
            "informational_needs": [
                "Bulk, machine-readable data",
                "Consistent labels and categorization",
                "Historical depth and trends",
                "Metadata completeness",
                "Standardized formatting",
                "Research methodology support"
            ],
            "key_evaluation_dimensions": [
                "Metadata completeness",
                "Machine readability",
                "Consistency across documents",
                "Data structure quality",
                "Research utility"
            ],
            "mission_alignment": "Clearinghouse exposes a public API for data extraction, welcoming usage by researchers as mentioned in founding literature.",
            "usage": "Academic and data-science researchers will use the legal summary as an input for their models to conduct large-scale analysis of case outcomes and legal trends."
        }
}




# Create Persona instances from the JSON data
LEGAL_DOCUMENT_PERSONAS = {}
for persona_key, persona_data in RAW_LEGAL_DOCUMENT_PERSONAS.items():
    LEGAL_DOCUMENT_PERSONAS[persona_key] = Persona(
        name=persona_data["name"],
        description=persona_data["description"],
        representative_users=persona_data["representative_users"],
        informational_needs=persona_data["informational_needs"],
        key_evaluation_dimensions=persona_data["key_evaluation_dimensions"],
        mission_alignment=persona_data["mission_alignment"],
        usage=persona_data["usage"]
    )


def get_persona(persona_name: str) -> Optional[Persona]:
    """Get a persona by name."""
    return LEGAL_DOCUMENT_PERSONAS.get(persona_name)


def get_all_personas() -> Dict[str, Persona]:
    """Get all available personas."""
    return LEGAL_DOCUMENT_PERSONAS.copy()


def get_persona_names() -> List[str]:
    """Get list of all persona names."""
    return list(LEGAL_DOCUMENT_PERSONAS.keys())

