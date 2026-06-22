#!/usr/bin/env bash
set -euo pipefail

# Install the Psi0 Jetson Thor Python environment.
#
# This script intentionally uses the Thor-specific pyproject.toml and uv.lock
# under scripts/deployment/thor, instead of the root Psi0 pyproject.toml.
#
# Usage:
#   bash scripts/deployment/thor/install_deps.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV_DIR="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv-psi}"
VENV_PYTHON="$VENV_DIR/bin/python"

echo "[Psi0 Thor] Repo root: $REPO_ROOT"
echo "[Psi0 Thor] Thor project: $SCRIPT_DIR"
echo "[Psi0 Thor] Virtualenv: $VENV_DIR"

cd "$REPO_ROOT"

echo "[Psi0 Thor] Syncing locked Thor dependencies..."
UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync \
  --project "$SCRIPT_DIR" \
  --no-install-project \
  --locked

echo "[Psi0 Thor] Installing LeRobot without dependencies..."
echo "[Psi0 Thor] Reason: lerobot==0.3.3 metadata conflicts with the known-working Thor torchvision stack."
uv pip install \
  --python "$VENV_PYTHON" \
  --no-deps \
  "lerobot==0.3.3"

echo "[Psi0 Thor] Installing local Psi0 source editable without dependencies..."
echo "[Psi0 Thor] Reason: root pyproject has non-Thor dependency constraints; Thor deps are managed above."
uv pip install \
  --python "$VENV_PYTHON" \
  --no-deps \
  -e "$REPO_ROOT"

echo "[Psi0 Thor] Done."
echo
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
