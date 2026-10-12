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

python3 -m pip install uv==0.8.22
python_bin=python3
if ! python3 -c 'import sqlite3' >/dev/null 2>&1; then
  # The Workers Build Python image omits _sqlite3. Use python-build-standalone,
  # which includes the standard library extension used by these tests on CI.
  uv python install 3.11
  uv venv --python 3.11 .venv
  python_bin=.venv/bin/python
fi
uv pip install --python "$python_bin" -r requirements.txt pytest pytest-cov
npm install --no-audit --no-fund
# Workers Builds has no privileged package manager. Install pinned static
# binaries as temporary test dependencies instead of using apt or sudo.
npm install --no-save --no-package-lock ffmpeg-static@5.2.0 ffprobe-static@3.1.0
ffmpeg_bin="$(node -p "require('ffmpeg-static')")"
ffprobe_bin="$(node -p "require('ffprobe-static').path")"
export PATH="$(dirname "$ffmpeg_bin"):$(dirname "$ffprobe_bin"):$PATH"
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null

# Use the PostgreSQL 16 service from the existing GitHub test job. Workers
# Builds provides Docker; this container is isolated and removed on exit.
pg_container="speakr-ci-postgres-${WORKERS_CI_BUILD_UUID:?Missing Workers CI build UUID}"
stop_postgres() {
  docker stop "$pg_container" >/dev/null 2>&1 || true
}
trap stop_postgres EXIT
docker run --detach --rm --name "$pg_container" \
  --cgroupns=host \
  --publish 127.0.0.1:5432:5432 \
  --env POSTGRES_USER=speakr \
  --env POSTGRES_PASSWORD=speakr \
  --env POSTGRES_DB=speakr_test \
  postgres:16
for attempt in $(seq 1 60); do
  if docker exec "$pg_container" pg_isready -U speakr -d speakr_test >/dev/null 2>&1; then
    break
  fi
  if [ "$attempt" -eq 60 ]; then
    echo "PostgreSQL service did not become ready" >&2
    exit 1
  fi
  sleep 1
done
export TEST_DATABASE_URI=postgresql://speakr:speakr@127.0.0.1:5432/speakr_test
pytest_args=(tests/ -q --cov=src --cov-report=term-missing --cov-fail-under=54)
"$python_bin" -m pytest "${pytest_args[@]}"
npm test

