#!/usr/bin/env bash
# Option A: activate sibling insilico .venv (tribev2 + torch already installed).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/../insilico/.venv/bin/activate"
echo "Active venv: ${VIRTUAL_ENV}"
echo "Run Python from: ${ROOT}"
