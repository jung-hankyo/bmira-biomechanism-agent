#!/bin/bash
# Cloud sessions only: install B-MiRA with its test, lint and app extras so the
# offline tests (`pytest`) and the linter (`ruff check .`) run at once.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"
python -m pip install --quiet --disable-pip-version-check --root-user-action=ignore -e ".[dev,app]"
