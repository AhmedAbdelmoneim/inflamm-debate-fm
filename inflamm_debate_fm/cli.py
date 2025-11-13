"""Command-line interface for inflamm-debate-fm.

This module provides backward compatibility by importing the main CLI app.
New code should import from `inflamm_debate_fm.cli` instead.
"""

from inflamm_debate_fm.cli import app

__all__ = ["app"]

if __name__ == "__main__":
    app()
