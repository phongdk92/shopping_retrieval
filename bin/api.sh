#!/usr/bin/env bash
source bin/setvars.sh
export PROMETHEUS_PORT=8010
exec uvicorn --workers 1 \
             --host 0.0.0.0 \
             --port $1 \
             --log-config src/api/logging.yaml \
             --app-dir src/api app:app
