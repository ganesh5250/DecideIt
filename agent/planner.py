from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ProjectTask(BaseModel):
    """Structured input task for the DecideIt AI Agent (Section 3)."""

    project_title: Optional[str] = None
    problem_description: str = Field(..., min_length=5)
    proposed_solution: Optional[str] = None
    objective: Optional[str] = None
    country: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    target_beneficiaries: Optional[str] = None
    budget: Optional[str] = None
    timeline: Optional[str] = None
    available_resources: Optional[str] = None
    technical_constraints: Optional[str] = None
    expected_outcomes: Optional[str] = None
    funding_requirements: Optional[str] = None
    supporting_documents: List[str] = Field(default_factory=list)
    additional_requirements: Optional[str] = None

    @field_validator("problem_description")
    @classmethod
    def validate_problem_not_blank(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("problem_description cannot be empty or whitespace.")
        return cleaned

    @property
    def submission_mode(
        self,
    ) -> Literal["problem_only", "problem_and_solution", "problem_solution_and_docs", "problem_and_docs"]:
        has_solution = bool(self.proposed_solution and self.proposed_solution.strip())
        has_docs = bool(self.supporting_documents)
        if has_solution and has_docs:
            return "problem_solution_and_docs"
        if has_solution:
            return "problem_and_solution"
        if has_docs:
            return "problem_and_docs"
        return "problem_only"

    def missing_optional_fields(self) -> List[str]:
        """List optional fields that were not provided by the user."""
        optional_fields = [
            "project_title",
            "proposed_solution",
            "objective",
            "country",
            "state",
            "district",
            "target_beneficiaries",
            "budget",
            "timeline",
            "available_resources",
            "technical_constraints",
            "expected_outcomes",
            "funding_requirements",
            "additional_requirements",
        ]
        missing = []
        for field_name in optional_fields:
            val = getattr(self, field_name)
            if val is None or (isinstance(val, str) and not val.strip()):
                missing.append(field_name)
        if not self.supporting_documents:
            missing.append("supporting_documents")
        return missing


class WorkflowStage(BaseModel):
    """Represents a single stage in the planned evaluation workflow."""

    stage_id: int
    name: str
    description: str
    required: bool = True
    skip_reason: Optional[str] = None
    recommended_tool: Optional[str] = None


class WorkflowPlan(BaseModel):
    """Complete task-specific workflow plan."""

    submission_mode: str
    missing_fields: List[str]
    stages: List[WorkflowStage]


class Planner:
    """Builds a task-specific workflow based on the submitted ProjectTask (Section 6.2)."""

    def create_plan(
        self, task: ProjectTask, available_tools: Optional[List[str]] = None
    ) -> WorkflowPlan:
        tools_set = set(available_tools or [])
        has_docs = bool(task.supporting_documents)
        has_solution = bool(task.proposed_solution and task.proposed_solution.strip())
        has_location_or_funding_context = bool(
            (task.country and task.country.strip())
            or (task.funding_requirements and task.funding_requirements.strip())
            or (task.objective and task.objective.strip())
        )

        stages: List[WorkflowStage] = [
            WorkflowStage(
                stage_id=1,
                name="validate_input",
                description="Validate the task input and identify missing optional fields.",
                required=True,
            ),
            WorkflowStage(
                stage_id=2,
                name="extract_documents",
                description="Read and extract information from supporting documents.",
                required=has_docs,
                skip_reason=None if has_docs else "No supporting_documents provided in task.",
                recommended_tool="document_reader" if has_docs else None,
            ),
            WorkflowStage(
                stage_id=3,
                name="extract_requirements",
                description="Extract project requirements, objectives, and constraints.",
                required=True,
            ),
            WorkflowStage(
                stage_id=4,
                name="analyse_problem",
                description="Analyse the submitted problem, root causes, and local context.",
                required=True,
            ),
            WorkflowStage(
                stage_id=5,
                name="analyse_or_generate_solution",
                description=(
                    "Analyse the proposed solution and assess alignment with the problem."
                    if has_solution
                    else "Generate possible solution approaches because no proposed_solution was supplied."
                ),
                required=True,
            ),
            WorkflowStage(
                stage_id=6,
                name="identify_gaps_and_assumptions",
                description="Identify missing information, contradictions, and underlying assumptions.",
                required=True,
            ),
            WorkflowStage(
                stage_id=7,
                name="retrieve_research_and_evidence",
                description="Retrieve relevant research, standards, and supporting evidence.",
                required=("research_search" in tools_set or "web_search" in tools_set),
                skip_reason=(
                    None
                    if ("research_search" in tools_set or "web_search" in tools_set)
                    else "No research_search or web_search tools configured."
                ),
                recommended_tool="research_search",
            ),
            WorkflowStage(
                stage_id=8,
                name="evaluate_environmental_impact",
                description="Evaluate environmental impact and relevant environmental indicators.",
                required=True,
                recommended_tool="environmental_analysis" if "environmental_analysis" in tools_set else None,
            ),
            WorkflowStage(
                stage_id=9,
                name="evaluate_social_impact",
                description="Evaluate social impact, beneficiaries, and distributional effects.",
                required=True,
                recommended_tool="social_analysis" if "social_analysis" in tools_set else None,
            ),
            WorkflowStage(
                stage_id=10,
                name="assess_feasibility",
                description="Assess technical, operational, and financial feasibility.",
                required=True,
                recommended_tool="feasibility_analysis" if "feasibility_analysis" in tools_set else None,
            ),
            WorkflowStage(
                stage_id=11,
                name="identify_risks_and_mitigation",
                description="Identify technical, social, environmental, and operational risks and mitigation options.",
                required=True,
            ),
            WorkflowStage(
                stage_id=12,
                name="search_funding_opportunities",
                description="Search and evaluate potentially relevant funding programmes.",
                required=has_location_or_funding_context or ("funding_search" in tools_set),
                skip_reason=(
                    None
                    if (has_location_or_funding_context or "funding_search" in tools_set)
                    else "Insufficient geographic/project details and no funding_search tool configured."
                ),
                recommended_tool="funding_search" if "funding_search" in tools_set else None,
            ),
            WorkflowStage(
                stage_id=13,
                name="validate_and_assemble_findings",
                description="Validate claims, evidence labels, and assemble evaluation findings.",
                required=True,
            ),
            WorkflowStage(
                stage_id=14,
                name="generate_report",
                description="Generate the final structured evaluation report.",
                required=True,
                recommended_tool="report_generator" if "report_generator" in tools_set else None,
            ),
        ]

        return WorkflowPlan(
            submission_mode=task.submission_mode,
            missing_fields=task.missing_optional_fields(),
            stages=stages,
        )