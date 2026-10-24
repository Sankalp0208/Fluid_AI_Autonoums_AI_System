import logging
from typing import Tuple
from app.config import settings
from app.core.llm_factory import llm_manager
from app.schemas import ExecutionPlan, DocumentDraft, ReflectionFeedback

logger = logging.getLogger("agent.reflection")

REFLECTION_SYSTEM_PROMPT = """
You are an Executive Review Board Member and Quality Auditor.
Your responsibility is to rigorously audit the drafted document against the user's initial request and the strategic execution plan.

Evaluation Criteria:
1. Completeness: Were all components of the user's request and the planned tasks fulfilled?
2. Ambiguity Handling: Were implicit constraints, missing data, and tradeoffs effectively resolved?
3. Actionability & Depth: Does the content contain concrete specifics (timelines, architecture choices, metrics, risk mitigation) rather than superficial filler?
4. Formatting & Executive Presence: Are sections well-structured, containing tables, bullet points, and highlighted executive callouts?

Score the document from 1 to 10:
- 8-10: Publication-ready, comprehensive, deep, and decisive.
- 6-7: Acceptable but contains minor gaps in depth or missing quantitative metrics.
- 1-5: Incomplete, superficial, or fails to address primary requirements.

Provide strengths, constructive critiques, and actionable revisions.
"""

REVISION_SYSTEM_PROMPT = """
You are an Executive Revision Specialist.
Your mission is to take the drafted document and the Quality Auditor's critique and actionable revisions, and refine the document to perfection.
Enhance technical depth, add missing metrics or tables, and polish executive tone.
"""

class AgentReflector:
    """Evaluates the drafted document and applies iterative self-correction if quality standards are not met."""

    def __init__(self):
        self.llm = llm_manager
        self.min_acceptable_score = settings.MIN_ACCEPTABLE_SCORE
        self.max_reflection_rounds = settings.MAX_REFLECTION_ROUNDS

    def evaluate_and_refine(
        self,
        user_request: str,
        plan: ExecutionPlan,
        draft: DocumentDraft
    ) -> Tuple[DocumentDraft, ReflectionFeedback]:
        """
        Conducts self-check reflection on the draft. If score is below threshold,
        triggers automatic revision loop.
        """
        logger.info("Conducting autonomous Reflection & Self-Check...")

        # 1. Evaluate draft
        feedback = self._critique_draft(user_request, plan, draft)
        logger.info(f"Reflection completed. Score: {feedback.score}/10. Acceptable: {feedback.is_acceptable}")

        # 2. Iterative Self-Correction Loop if below standard
        current_draft = draft
        round_count = 0

        while (feedback.score < self.min_acceptable_score or not feedback.is_acceptable) and (round_count < self.max_reflection_rounds):
            round_count += 1
            logger.warning(f"Draft scored {feedback.score}/10 (Threshold: {self.min_acceptable_score}). Initiating Self-Correction Round {round_count}...")
            
            current_draft = self._apply_revisions(user_request, plan, current_draft, feedback)
            feedback = self._critique_draft(user_request, plan, current_draft)
            feedback.revisions_applied = True
            logger.info(f"Post-revision score: {feedback.score}/10.")

        return current_draft, feedback

    def _critique_draft(self, user_request: str, plan: ExecutionPlan, draft: DocumentDraft) -> ReflectionFeedback:
        sections_summary = "\n".join([
            f"- Section '{s.title}': {len(s.content)} chars, {len(s.bullet_points or [])} bullets, "
            f"Table={'Yes' if s.table else 'No'}, Callout={'Yes' if s.callout_box else 'No'}"
            for s in draft.sections
        ])

        user_prompt = f"""Review the following document deliverable:
Original User Request: "{user_request}"
Document Title: "{draft.title}"
Document Type: "{draft.document_type}"
Execution Plan Tasks: {[t.title for t in plan.tasks]}
Assumptions Used: {plan.assumptions}

Executive Summary:
"{draft.executive_summary}"

Section Structure:
{sections_summary}

Full Draft Content Sample:
{" ".join([s.content[:300] for s in draft.sections])}

Perform a rigorous evaluation and output ReflectionFeedback matching the schema."""

        feedback: ReflectionFeedback = self.llm.generate_structured(
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_class=ReflectionFeedback
        )
        return feedback

    def _apply_revisions(
        self,
        user_request: str,
        plan: ExecutionPlan,
        draft: DocumentDraft,
        feedback: ReflectionFeedback
    ) -> DocumentDraft:
        user_prompt = f"""Original User Request: "{user_request}"
Document Title: "{draft.title}"

Auditor Critique:
{chr(10).join(f"- {c}" for c in feedback.critique)}

Required Actionable Revisions:
{chr(10).join(f"- {r}" for r in feedback.actionable_revisions)}

Current Draft Sections:
{[s.title for s in draft.sections]}

Refine and elevate the entire DocumentDraft to resolve all critiques and incorporate all actionable revisions."""

        revised_draft: DocumentDraft = self.llm.generate_structured(
            system_prompt=REVISION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema_class=DocumentDraft
        )
        return revised_draft

