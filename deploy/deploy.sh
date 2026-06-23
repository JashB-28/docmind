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

# Compose reads its interpolation .env from the compose file's directory
# (deploy/), not the repo root — so the root .env's PROXY_NETWORK / DOCMIND_IMAGE
# aren't seen for ${...} substitution. Export them explicitly from the root .env
# (shell env wins in interpolation). The service's runtime env still comes from
# the compose `env_file: ../.env`.
if [ -f .env ]; then
  export PROXY_NETWORK="$(grep -E '^PROXY_NETWORK=' .env | head -1 | cut -d= -f2-)"
  export DOCMIND_IMAGE="$(grep -E '^DOCMIND_IMAGE=' .env | head -1 | cut -d= -f2-)"
fi

docker compose -f "$COMPOSE_FILE" pull
docker compose -f "$COMPOSE_FILE" up -d
docker image prune -f
echo "Deployed ${DOCMIND_IMAGE:-jash09/docmind:latest} via $COMPOSE_FILE"
