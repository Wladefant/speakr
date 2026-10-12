#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

python3 - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Workers Builds requires Python 3.11; found {sys.version.split()[0]}")
PY

node_major="$(node --version)"
case "$node_major" in
  v20.*) ;;
  *) echo "Workers Builds requires Node.js 20; found ${node_major}" >&2; exit 1 ;;
esac

python_bin=python3
if ! python3 -c 'import sqlite3' >/dev/null 2>&1; then
  # The Workers Build Python image omits _sqlite3. Use python-build-standalone,
  # which includes the standard library extension used by these tests on CI.
  python3 -m pip install uv==0.8.22
  uv python install 3.11
  uv venv --python 3.11 .venv
  python_bin=.venv/bin/python
fi
uv pip install --python "$python_bin" -r requirements.txt pytest pytest-cov
npm install --no-audit --no-fund

# Workers Builds has no privileged package manager. Install pinned static
# binaries as temporary test dependencies instead of using apt or sudo.
pytest_args=(tests/ -q --ignore=tests/test_postgres_migrations.py --cov=src --cov-report=term-missing --cov-fail-under=54)
"$python_bin" -m pytest "${pytest_args[@]}"
npm test

