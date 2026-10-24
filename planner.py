import logging
from app.core.llm_factory import llm_manager
from app.schemas import ExecutionPlan

logger = logging.getLogger("agent.planner")

PLANNER_SYSTEM_PROMPT = """
You are an elite Autonomous AI Strategic Planner, Enterprise Solutions Architect, and Principal Technical Analyst.
Your role is to deeply understand the user's exact request and formulate an autonomous, highly specific, multi-step execution plan to produce an authoritative, comprehensive document.

Domain & Document Genre Guidelines:
1. Exact Document Identification:
   - FRD (Functional Requirements Document): Formulate tasks covering System Scope, Specific Functional Requirements (FR-01, FR-02...), Multi-Step Process Workflows, Data Contracts/Tool Interfaces, and Non-Functional/Acceptance Criteria.
   - PRD (Product Requirements Document): Formulate tasks covering Product Vision, Target Personas, Core Features, API Integrations, and Release Roadmap.
   - SOP (Standard Operating Procedure): Formulate tasks covering Procedural Scope, RACI Responsibility Matrix, Step-by-Step Execution Protocols, Exception Handling, and Quality Audits.
   - Technical Architecture / System Design: Formulate tasks covering Component Architecture, Inter-Service Data Flows, API Contracts, Fault Tolerance, and Observability.
   - Executive Decision Paper / Proposal: Formulate tasks covering Business Problem, Option Analysis & Tradeoffs, Recommended Solution with Cost/ROI, Risk Mitigations, and Phased Timelines.

2. Deep Topic Specialization:
   - Adapt the tasks and objectives completely to the user's specific domain (e.g. if the user asks for a 'Multi-Agent Research System', tasks MUST explicitly address agent roles, orchestrator coordination, web search tools, context synthesis, fact-checking verification, and rate-limit handling).

3. Ambiguity & Constraint Resolution:
   - Formulate 3-5 explicit, realistic ASSUMPTIONS that resolve implicit constraints (budgets, technologies, SLAs, data privacy, concurrency limits).

4. Task Breakdown:
   - Deconstruct the deliverable into 4 to 6 logically ordered, concrete execution tasks. Each task must specify the exact technical, functional, or process analysis it will produce.
"""

class AgentPlanner:
    """Autonomous multi-step planner."""

    def __init__(self):
        self.llm = llm_manager

    def create_plan(self, user_request: str) -> ExecutionPlan:
        """
        Analyzes the user's natural language request and generates a tailored ExecutionPlan.
        """
        logger.info(f"Formulating autonomous execution plan for: '{user_request[:80]}...'")
        user_prompt = f"""User Request:
"{user_request}"

Create an executive-level, structured execution plan specifically tailored to the requested topic and document type.
Provide:
- Document Title (specific to the user's topic)
- Document Type (e.g., FRD, PRD, SOP, Technical Design, Decision Paper)
- Target Audience
- Strategic Objective
- Realistic Domain Assumptions
- 4 to 6 Ordered Execution Tasks with actionable descriptions detailing the exact functional requirements, processes, or analysis each task covers."""

        plan: ExecutionPlan = self.llm.generate_structured(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_class=ExecutionPlan
        )
        logger.info(f"Plan formulated successfully: '{plan.document_title}' ({plan.document_type}) with {len(plan.tasks)} tasks.")
        return plan
