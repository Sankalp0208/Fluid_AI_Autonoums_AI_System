import logging
import os
import time
from app.agent.executor import AgentExecutor
from app.agent.planner import AgentPlanner
from app.agent.reflection import AgentReflector
from app.config import settings
from app.schemas import AgentRequest, AgentResponse
from app.tools.docx_generator import docx_generator

logger = logging.getLogger("agent.orchestrator")

class AutonomousDocumentAgent:
    """
    End-to-end Autonomous Agent orchestrating:
    1. Dynamic Multi-Step Planning & Assumption Generation
    2. Sequential Task Execution & Content Synthesis
    3. Self-Check Reflection & Automatic Revision Loop
    4. Publication-Grade Word (.docx) Document Compilation
    """

    def __init__(self):
        self.planner = AgentPlanner()
        self.executor = AgentExecutor()
        self.reflector = AgentReflector()
        self.docx_tool = docx_generator

    def run(self, request_payload: AgentRequest, base_url: str = "") -> AgentResponse:
        start_time = time.time()
        user_request = request_payload.request
        logger.info(f"Initiating autonomous agent workflow for request: '{user_request[:80]}...'")

        # Step 1: Autonomous Multi-Step Planning
        plan = self.planner.create_plan(user_request)

        # Step 2: Task Execution & Draft Generation
        draft = self.executor.execute_plan(user_request, plan)

        # Step 3: Reflection & Self-Check / Revision Loop (Mandatory Engineering Improvement)
        if settings.ENABLE_REFLECTION:
            final_draft, reflection_feedback = self.reflector.evaluate_and_refine(
                user_request=user_request,
                plan=plan,
                draft=draft
            )
        else:
            final_draft = draft
            from app.schemas import ReflectionFeedback
            reflection_feedback = ReflectionFeedback(
                score=8,
                is_acceptable=True,
                completeness_assessment="Reflection disabled; accepted initial draft.",
                strengths=["Execution completed successfully"],
                critique=[],
                actionable_revisions=[]
            )

        # Step 4: Polished Microsoft Word (.docx) Generation
        file_path = self.docx_tool.create_document(final_draft)
        file_name = os.path.basename(file_path)

        # Build download URL
        clean_base = base_url.rstrip("/") if base_url else ""
        download_url = f"{clean_base}/documents/{file_name}" if clean_base else f"/documents/{file_name}"

        elapsed_time = round(time.time() - start_time, 2)
        logger.info(f"Agent finished in {elapsed_time}s. Output generated: {file_path}")

        return AgentResponse(
            success=True,
            document_title=final_draft.title,
            document_type=final_draft.document_type,
            file_name=file_name,
            file_path=file_path,
            download_url=download_url,
            execution_plan=plan,
            reflection=reflection_feedback,
            summary=final_draft.executive_summary,
            execution_time_seconds=elapsed_time
        )


autonomous_agent = AutonomousDocumentAgent()

