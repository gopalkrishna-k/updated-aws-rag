"""Main Entry Point for AWS Overview Multi-Stage Hybrid RAG System.

Runs the interactive question-answering REPL:
- Input via input("> ")
- Displays formatted answer and citations
- Displays retrieved chunks sent to LLM
- Tracks and displays total duration for each query
- Separates Q&A sessions with clean line dividers
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import main as cli_main

if __name__ == "__main__":
    sys.exit(cli_main())

