from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionTask(BaseModel):
    id: int
    title: str = Field(..., description="Short title of the task")
    description: str = Field(..., description="Detailed description of what will be performed")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    result_summary: Optional[str] = Field(default=None, description="Summary of step execution result")


class ExecutionPlan(BaseModel):
    document_title: str = Field(..., description="Professional title of the document to produce")
    document_type: str = Field(..., description="Type of document, e.g., PRD, SOP, Decision Paper, Proposal")
    target_audience: str = Field(..., description="Audience, e.g., C-Suite, Engineering Team, Operations")
    objective: str = Field(..., description="Core business/technical goal of this document")
    assumptions: List[str] = Field(default_factory=list, description="Reasonable assumptions made to resolve ambiguities")
    tasks: List[ExecutionTask] = Field(..., description="Ordered list of tasks to execute")


class TableData(BaseModel):
    headers: List[str]
    rows: List[List[str]]


class DocumentSection(BaseModel):
    title: str
    level: int = 1
    content: str
    bullet_points: Optional[List[str]] = None
    table: Optional[TableData] = None
    callout_box: Optional[str] = None


class DocumentDraft(BaseModel):
    title: str
    subtitle: Optional[str] = None
    document_type: str
    executive_summary: str
    metadata: Dict[str, str] = Field(default_factory=dict)
    sections: List[DocumentSection]


class ReflectionFeedback(BaseModel):
    score: int = Field(..., ge=1, le=10, description="Overall quality score from 1 to 10")
    is_acceptable: bool = Field(..., description="Whether the draft meets executive quality standards")
    completeness_assessment: str = Field(..., description="Check if all required sections & assumptions were met")
    strengths: List[str] = Field(default_factory=list)
    critique: List[str] = Field(default_factory=list)
    actionable_revisions: List[str] = Field(default_factory=list)
    revisions_applied: bool = False


class AgentRequest(BaseModel):
    request: str = Field(
        ...,
        min_length=5,
        description="Natural language request for the document/deliverable to create",
        examples=[
            "Create a comprehensive Product Requirements Document (PRD) for an AI-Powered Customer Support Automation System."
        ]
    )


class AgentResponse(BaseModel):
    success: bool
    document_title: str
    document_type: str
    file_name: str
    file_path: str
    download_url: str
    execution_plan: ExecutionPlan
    reflection: ReflectionFeedback
    summary: str
    execution_time_seconds: float

