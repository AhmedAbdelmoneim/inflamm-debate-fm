"""Main CLI application and command registration."""

import typer

from inflamm_debate_fm.cli import analyze, embed, finetune, plot, preprocess, probe

app = typer.Typer(help="inflamm-debate-fm CLI")

# Register subcommands
app.add_typer(preprocess.app, name="preprocess")
app.add_typer(embed.app, name="embed")
app.add_typer(probe.app, name="probe")
app.add_typer(analyze.app, name="analyze")
app.add_typer(plot.app, name="plot")
app.add_typer(finetune.app, name="finetune")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
