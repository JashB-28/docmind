#!/usr/bin/env bash
# Runs ON the EC2 box (invoked by SSM from the deploy workflow): pulls the latest
# published image from Docker Hub and restarts the stack. Public image, so no
# registry login is needed.
set -euo pipefail

cd /opt/docmind

COMPOSE_FILE="${COMPOSE_FILE:-deploy/docker-compose.prod.yml}"
docker compose -f "$COMPOSE_FILE" pull
docker compose -f "$COMPOSE_FILE" up -d
docker image prune -f
echo "Deployed jash09/docmind:latest"
