#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
pg_bin=${HERASYS_PG_BIN:-/usr/lib/postgresql/16/bin}
pg_port=${HERASYS_PG_PORT:-15432}
pg_dir="$repo_dir/.runtime/postgres"
pg_socket="$repo_dir/.runtime/pgsocket"
mkdir -p "$pg_dir" "$pg_socket"
if [ ! -f "$pg_dir/PG_VERSION" ]; then
    "$pg_bin/initdb" -D "$pg_dir" --auth-local=trust --auth-host=trust --encoding=UTF8 --no-locale
fi
if ! "$pg_bin/pg_ctl" -D "$pg_dir" status; then
    "$pg_bin/pg_ctl" -D "$pg_dir" -l "$repo_dir/.runtime/postgres.log" \
        -o "-p $pg_port -h 127.0.0.1 -k $pg_socket" -w start
fi
printf 'export HERASYS_PG_DSN="host=127.0.0.1 port=%s dbname=postgres user=%s"\n' "$pg_port" "$(id -un)"
