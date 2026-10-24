"""
Pre-configured test inputs demonstrating the two required test scenarios:
1. Standard Business Request: Well-structured, explicit requirements (PRD).
2. Complex Ambiguous Request: Missing info, conflicting constraints, tight budget, requires autonomous assumptions & technical tradeoffs.
"""

TEST_CASES = {
    "standard": {
        "name": "Test Case 1: Standard Business Request (PRD)",
        "description": "Standard business request with well-defined sections and clear scope.",
        "request": (
            "Create a comprehensive Product Requirements Document (PRD) for an AI-Powered "
            "Customer Support Automation System. Include executive summary, target persona, "
            "key features, functional requirements, non-functional requirements, API integrations, "
            "and rollout phases."
        )
    },
    "complex": {
        "name": "Test Case 2: Complex Ambiguous Request (Decision Paper & Transition Plan)",
        "description": "Ambiguous request with missing data, conflicting requirements (tight budget vs major transition, holiday blackout vs Q4 deadline), and unresolved tech choices (SAP vs microservices).",
        "request": (
            "We need some sort of operational plan for transitioning our legacy retail inventory "
            "across 40 stores to an omnichannel warehouse model by Q4. Budget is tight, store "
            "managers are worried about stock disruptions during holiday season, and IT hasn't "
            "decided between SAP or custom microservices. Give me a complete executive decision "
            "paper and phased transition plan with mitigation strategies and recommended choice."
        )
    }
}

