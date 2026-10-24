import os
import pytest
from app.agent.orchestrator import autonomous_agent
from app.schemas import AgentRequest
from tests.test_cases import TEST_CASES


def test_standard_business_request():
    request_obj = AgentRequest(request=TEST_CASES["standard"]["request"])
    response = autonomous_agent.run(request_obj)
    
    assert response.success is True
    assert len(response.execution_plan.tasks) > 0
    assert response.file_path is not None
    assert os.path.exists(response.file_path)
    assert response.reflection.score >= 7
    assert len(response.execution_plan.assumptions) >= 0


def test_complex_ambiguous_request():
    request_obj = AgentRequest(request=TEST_CASES["complex"]["request"])
    response = autonomous_agent.run(request_obj)
    
    assert response.success is True
    assert len(response.execution_plan.assumptions) > 0
    assert os.path.exists(response.file_path)
    assert response.reflection.score >= 7

