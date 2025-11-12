"""Data processing pipeline command."""

import typer

from inflamm_debate_fm.data.pipeline import run_pipeline

app = typer.Typer(help="Data processing pipeline")


@app.command("data")
def process_data() -> None:
    """Run the complete data processing pipeline."""
    run_pipeline()
