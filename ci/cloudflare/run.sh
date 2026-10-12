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

# Install the prebuilt PostgreSQL 16 server in user space; Workers Docker
# cannot create containers in its cgroup namespace.
pg_root=".ci-postgres-${WORKERS_CI_BUILD_UUID:?Missing Workers CI build UUID}"
mkdir -p "$pg_root"
curl --fail --location --silent --show-error https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C "$pg_root" --strip-components=1 bin/micromamba
export MAMBA_ROOT_PREFIX="$PWD/$pg_root/mamba-root"
"$pg_root/micromamba" create --yes --prefix "$PWD/$pg_root/env" \
  --channel conda-forge "postgresql=16"
pg_bin="$pg_root/env/bin"
pg_data="$pg_root/data"
mkdir -p "$pg_data"
stop_postgres() {
  "$pg_bin/pg_ctl" -D "$pg_data" -m immediate stop >/dev/null 2>&1 || true
}
trap stop_postgres EXIT
"$pg_bin/initdb" -D "$pg_data" -U speakr --auth-local=trust --auth-host=trust
"$pg_bin/pg_ctl" -D "$pg_data" -l "$pg_root/postgres.log" \
  -o "-h 127.0.0.1 -p 5432" -w start
"$pg_bin/createdb" -h 127.0.0.1 -p 5432 -U speakr speakr_test
for attempt in $(seq 1 60); do
  if "$pg_bin/pg_isready" -h 127.0.0.1 -p 5432 -U speakr -d speakr_test >/dev/null 2>&1; then
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

