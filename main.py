import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.agent.orchestrator import autonomous_agent
from app.config import settings
from app.schemas import AgentRequest, AgentResponse

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Autonomous AI Agent with Multi-Step Planning, Self-Check Reflection, Model Fallback, and Word (.docx) Generation."
)

# Enable CORS for frontend or external client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "endpoints": {
            "agent": "POST /agent",
            "health": "GET /health",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "healthy",
        "primary_model": settings.PRIMARY_MODEL,
        "fallback_model": settings.FALLBACK_MODEL,
        "groq_configured": bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY.strip()),
        "output_directory": settings.OUTPUT_DIR,
        "reflection_enabled": settings.ENABLE_REFLECTION
    }


@app.post("/agent", response_model=AgentResponse, tags=["Autonomous Agent"])
def run_agent_endpoint(payload: AgentRequest, request: Request):
    """
    Accepts natural language request, determines required tasks, formulates plan,
    executes steps, conducts self-reflection/critique, and generates Word document.
    """
    try:
        base_url = str(request.base_url).rstrip("/")
        response = autonomous_agent.run(payload, base_url=base_url)
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Autonomous Agent execution failed: {str(e)}")


@app.get("/documents/{filename}", tags=["Documents"])
def download_document(filename: str):
    """
    Download endpoint for generated Microsoft Word (.docx) documents.
    """
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(settings.OUTPUT_DIR, safe_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Requested document not found.")

    return FileResponse(
        path=file_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=safe_filename
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

