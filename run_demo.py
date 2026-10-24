import os
import sys
import time

# Reconfigure stdout for UTF-8 compatibility on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from app.agent.orchestrator import autonomous_agent
from app.config import settings
from app.schemas import AgentRequest
from tests.test_cases import TEST_CASES

console = Console(safe_box=True)

def run_test_scenario(scenario_key: str):
    case = TEST_CASES[scenario_key]
    console.print()
    console.print(Panel(
        f"[bold cyan]{case['name']}[/bold cyan]\n[dim]{case['description']}[/dim]",
        title="[bold yellow]TEST SCENARIO[/bold yellow]",
        border_style="cyan",
        box=box.ROUNDED
    ))

    console.print(f"[bold green]User Input Prompt:[/bold green]\n\"{case['request']}\"\n")

    with console.status("[bold green]Agent Thinking, Planning & Executing Steps...", spinner="dots"):
        request_obj = AgentRequest(request=case["request"])
        start_time = time.time()
        response = autonomous_agent.run(request_obj)
        elapsed = round(time.time() - start_time, 2)

    # 1. Display Plan & Generated Tasks
    console.print(f"\n[bold magenta]1. Dynamic Autonomous Plan Formulated:[/bold magenta]")
    console.print(f"   [bold]Document Title:[/bold] {response.document_title}")
    console.print(f"   [bold]Document Type:[/bold]  {response.document_type}")
    console.print(f"   [bold]Target Audience:[/bold] {response.execution_plan.target_audience}")
    console.print(f"   [bold]Objective:[/bold]       {response.execution_plan.objective}")

    # Display Assumptions
    if response.execution_plan.assumptions:
        console.print(f"\n   [bold yellow]Autonomous Assumptions Made to Resolve Ambiguities:[/bold yellow]")
        for a in response.execution_plan.assumptions:
            console.print(f"   * {a}")

    # Display Task Table
    task_table = Table(title="Generated Task Execution List", box=box.SIMPLE_HEAVY, show_header=True)
    task_table.add_column("Task ID", style="cyan", justify="center", width=8)
    task_table.add_column("Task Title", style="bold white", width=35)
    task_table.add_column("Status", style="green", width=12)
    task_table.add_column("Execution Summary", style="dim", width=45)

    for task in response.execution_plan.tasks:
        task_table.add_row(
            str(task.id),
            task.title,
            f"[{'green' if task.status == 'completed' else 'yellow'}]{task.status.value.upper()}[/]",
            task.result_summary or "Executed"
        )
    console.print(task_table)

    # 2. Display Reflection / Self-Check Results
    ref = response.reflection
    console.print(f"[bold magenta]2. Reflection & Self-Check Evaluation:[/bold magenta]")
    console.print(f"   [bold]Quality Score:[/bold] [{ 'green' if ref.score >= 8 else 'yellow' }]{ref.score}/10[/] (Acceptable: {ref.is_acceptable})")
    console.print(f"   [bold]Completeness:[/bold]  {ref.completeness_assessment}")
    if ref.strengths:
        console.print(f"   [bold]Strengths:[/bold]     {', '.join(ref.strengths[:2])}")
    if ref.critique:
        console.print(f"   [bold]Critique:[/bold]      {', '.join(ref.critique[:2])}")

    # 3. Document Output Deliverable
    console.print(f"\n[bold magenta]3. Final Microsoft Word Deliverable:[/bold magenta]")
    file_size_kb = round(os.path.getsize(response.file_path) / 1024, 1)
    console.print(Panel(
        f"[bold green]File Successfully Generated![/bold green]\n"
        f"* [bold]Local File Path:[/bold] [underline]{response.file_path}[/underline]\n"
        f"* [bold]File Size:[/bold]       {file_size_kb} KB\n"
        f"* [bold]Execution Time:[/bold]  {elapsed} seconds\n"
        f"* [bold]Download URL:[/bold]    {response.download_url}",
        title="[bold green]OUTPUT DELIVERABLE (.DOCX)[/bold green]",
        border_style="green",
        box=box.ROUNDED
    ))
    return response

def main():
    console.print(Panel.fit(
        "[bold white on blue] AUTONOMOUS AGENT 60-MINUTE BUILD CHALLENGE [/bold white on blue]\n"
        "[dim]FastAPI | LangChain | Groq LLM | Reflection/Self-Check | Retry/Fallback | Word (.docx) Engine[/dim]",
        border_style="blue"
    ))

    console.print("\n[bold]Demonstrating 2 Test Scenarios Required by Assignment Specification:[/bold]")
    console.print("1. Standard Business Request")
    console.print("2. Complex Ambiguous Request\n")

    # Run Standard Business Request
    console.print("=" * 80)
    console.print("[bold yellow]RUNNING TEST CASE 1: STANDARD BUSINESS REQUEST[/bold yellow]")
    console.print("=" * 80)
    run_test_scenario("standard")

    console.print("\n" + "=" * 80)
    console.print("[bold yellow]RUNNING TEST CASE 2: COMPLEX AMBIGUOUS REQUEST[/bold yellow]")
    console.print("=" * 80)
    run_test_scenario("complex")

    console.print("\n[bold green][OK] All test scenarios executed successfully![/bold green]")
    console.print("[cyan]To launch the FastAPI server, run:[/cyan] [bold]uvicorn app.main:app --reload --port 8000[/bold]\n")

if __name__ == "__main__":
    main()

