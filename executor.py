import logging
from typing import List
from app.core.llm_factory import llm_manager
from app.schemas import ExecutionPlan, TaskStatus, DocumentDraft, DocumentSection, TableData

logger = logging.getLogger("agent.executor")

EXECUTOR_SYSTEM_PROMPT = """
You are a Principal Enterprise Systems Architect, Lead Technical Writer, and Senior Business Consultant.
Your mission is to execute a strategic plan and author an authoritative, publication-grade Microsoft Word document tailored specifically and accurately to the user's prompt.

Technical Rigor & Structure Guidelines:
1. Exact Document Type Alignment:
   - FRD (Functional Requirements Document): Provide exhaustive functional specifications (e.g., FR-01, FR-02, FR-03), system actors/personas, end-to-end process workflows, tool integrations, data exchange contracts, and acceptance criteria.
   - PRD (Product Requirements Document): Provide user personas, feature priorities, UX user journeys, API matrices, and release milestones.
   - SOP (Standard Operating Procedure): Provide step-by-step procedural workflows, role responsibilities, exception protocols, and verification checkpoints.
   - Architecture Spec: Detail component boundaries, inter-service protocols, data storage schemas, and security controls.

2. Deep Domain Specificity:
   - If the request is about 'Multi-Agent Systems' or AI agents: Explicitly specify individual agent personas (e.g. Master Orchestrator, Deep Researcher, Synthesizer, Fact-Checking Evaluator), inter-agent message buses, shared memory state, tool calling interfaces (web search, vector databases, scraping engines), and loop termination conditions.
   - Write concrete, high-density technical and operational content. Never use vague generalities or shallow placeholders.

3. Structured Visual Elements:
   - Provide comprehensive TABLES for every major section (e.g., Functional Requirements Matrix, Process Sequence Steps, Tool & API Interface Schemas, RACI Matrix, or SLA Thresholds).
   - Use CALLOUT BOXES for critical architectural decisions, security warnings, operational mandates, or governance rules.
   - Use structured BULLET POINTS for detailed specifications, parameters, inputs, and outputs.

4. Executive Summary:
   - Include a compelling executive summary articulating the strategic mandate, architectural approach, and core capabilities.
"""

class AgentExecutor:
    """Executes the planned tasks and drafts the publication-grade document."""

    def __init__(self):
        self.llm = llm_manager

    def execute_plan(self, user_request: str, plan: ExecutionPlan) -> DocumentDraft:
        """
        Executes each planned task sequentially, updating status and generating the complete DocumentDraft.
        """
        logger.info(f"Executing plan for: '{plan.document_title}'")
        
        # 1. Update task statuses as in-progress
        for task in plan.tasks:
            task.status = TaskStatus.IN_PROGRESS

        # 2. Prepare comprehensive execution prompt
        tasks_text = "\n".join([f"- Task {t.id}: {t.title} -> {t.description}" for t in plan.tasks])
        assumptions_text = "\n".join([f"- {a}" for a in plan.assumptions]) if plan.assumptions else "Standard enterprise baseline."

        user_prompt = f"""Original User Request:
"{user_request}"

Execution Plan Context:
Document Title: {plan.document_title}
Document Type: {plan.document_type}
Target Audience: {plan.target_audience}
Strategic Objective: {plan.objective}

Assumptions to Incorporate:
{assumptions_text}

Tasks to Execute & Author:
{tasks_text}

Author the complete, detailed DocumentDraft matching the schema.
Ensure every section directly addresses the user's prompt with deep technical depth, functional requirements (FR-IDs where applicable), step-by-step processes, concrete comparison/specification tables, bullet points, and executive callout boxes."""

        draft: DocumentDraft = self.llm.generate_structured(
            system_prompt=EXECUTOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_class=DocumentDraft
        )

        # 3. Mark tasks as completed with summaries
        for idx, task in enumerate(plan.tasks):
            task.status = TaskStatus.COMPLETED
            if idx < len(draft.sections):
                task.result_summary = f"Authored '{draft.sections[idx].title}' with {len(draft.sections[idx].bullet_points or [])} action items and formatted structures."
            else:
                task.result_summary = f"Synthesized '{task.title}' into the core deliverable."

        logger.info(f"Execution finished. Generated {len(draft.sections)} document sections.")
        return draft
