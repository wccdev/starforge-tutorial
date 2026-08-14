#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

uv run lab migrate-v2
uv run ruff check nemo_rl_lab tests
uv run pytest -q
uv build .
