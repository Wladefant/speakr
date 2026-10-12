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

python3 -m pip install -r requirements.txt pytest pytest-cov
npm install --no-audit --no-fund

# Workers Builds has no privileged package manager. Install pinned static
# binaries as temporary test dependencies instead of using apt or sudo.
npm install --no-save --no-package-lock ffmpeg-static@5.2.0 ffprobe-static@3.1.0
ffmpeg_bin="$(node -p "require('ffmpeg-static')")"
ffprobe_bin="$(node -p "require('ffprobe-static').path")"
export PATH="$(dirname "$ffmpeg_bin"):$(dirname "$ffprobe_bin"):$PATH"
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null

# No PostgreSQL service is available in Workers Builds. Do not count or run
# the PostgreSQL migration integration module here.
python3 -m pytest tests/ -q \
  --ignore=tests/test_postgres_migrations.py \
  --cov=src --cov-report=term-missing --cov-fail-under=54
npm test

