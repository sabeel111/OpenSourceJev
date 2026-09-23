#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv "$PROJECT_ROOT/.venv"
fi

echo "Checking Python dependencies..."
"$VENV_PYTHON" -m pip install -r "$PROJECT_ROOT/requirements.txt" --disable-pip-version-check

echo "Starting OpenSourceJev at http://127.0.0.1:8000"
exec "$VENV_PYTHON" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
