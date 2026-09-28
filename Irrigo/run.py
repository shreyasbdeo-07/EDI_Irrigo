"""
Irrigo — Top-level server launcher
====================================
Run from the Irrigo/ directory:

    py -3 run.py

This starts the FastAPI backend on http://localhost:8000
Auto-reload is enabled for development.

Swagger UI  : http://localhost:8000/docs
ReDoc       : http://localhost:8000/redoc
Health      : http://localhost:8000/health
"""

import subprocess
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

subprocess.run(
    [
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ],
    cwd=str(HERE),
)
