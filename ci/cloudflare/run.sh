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

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then
      apt-get update
      apt-get install -y ffmpeg
    else
      sudo apt-get update
      sudo apt-get install -y ffmpeg
    fi
  else
    echo "ffmpeg and ffprobe are required by the Python test suite" >&2
    exit 1
  fi
fi

python3 -m pip install -r requirements.txt pytest pytest-cov
npm install --no-audit --no-fund

# This module targets PostgreSQL, which Workers Builds does not provide as a
# service. The SQLite-only schema tests in that module are not a substitute.
python3 -m pytest tests/ -q \
  --ignore=tests/test_postgres_migrations.py \
  --cov=src --cov-report=term-missing --cov-fail-under=54
npm test
