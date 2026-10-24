class AgentException(Exception):
    """Base exception for the autonomous agent system."""
    pass


class PlanningError(AgentException):
    """Raised when the agent fails to formulate a plan."""
    pass


class ExecutionError(AgentException):
    """Raised during the task execution step."""
    pass


class ReflectionError(AgentException):
    """Raised during self-reflection and validation."""
    pass


class DocumentGenerationError(AgentException):
    """Raised when docx generation fails."""
    pass


class LLMServiceUnavailableError(AgentException):
    """Raised when all LLM models and retries have been exhausted."""
    pass

