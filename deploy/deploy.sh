#!/usr/bin/env bash
# Runs ON the EC2 box (invoked by SSM from the deploy workflow): pulls the latest
# published image from Docker Hub and restarts the stack. Public image, so no
# registry login is needed.
set -euo pipefail

# Resolve the repo root from this script's own location, so it works no matter
# where the repo is cloned (~/docmind, /opt/docmind, ...). The earlier hardcoded
# `cd /opt/docmind` broke deploys on boxes that clone elsewhere.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Default to the behind-proxy layout (DocMind joins an existing proxy's network,
# publishes no host port) because that's how the live box runs. Override by
# exporting COMPOSE_FILE before invoking, e.g. for a standalone box that owns
# 80/443: COMPOSE_FILE=deploy/docker-compose.prod.yml bash deploy/deploy.sh
COMPOSE_FILE="${COMPOSE_FILE:-deploy/docker-compose.behind-proxy.yml}"

# PROXY_NETWORK / DOCMIND_IMAGE for the behind-proxy compose are read by Docker
# Compose from the repo-root .env automatically (it lives in $REPO_DIR == cwd).
docker compose -f "$COMPOSE_FILE" pull
docker compose -f "$COMPOSE_FILE" up -d
docker image prune -f
echo "Deployed ${DOCMIND_IMAGE:-jash09/docmind:latest} via $COMPOSE_FILE"
