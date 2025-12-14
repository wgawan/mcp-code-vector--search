#!/bin/bash

# Get the project root from environment or use current directory
PROJECT_ROOT="${PROJECT_ROOT:-.}"

# Get absolute path
PROJECT_ROOT=$(cd "$PROJECT_ROOT" && pwd)

# Persist directory (inside project)
PERSIST_DIR="${PROJECT_ROOT}/.vector-index"

# Create persist directory if it doesn't exist
mkdir -p "$PERSIST_DIR"

# Run Docker container with proper stdio handling
# Set DEBUG=1 environment variable to see stderr output for debugging
if [ "${DEBUG}" = "1" ]; then
  exec docker run --rm -i \
    -v "${PROJECT_ROOT}:/project:ro" \
    -v "${PERSIST_DIR}:/vector-index:rw" \
    -e PROJECT_ROOT=/project \
    -e PERSIST_DIR=/vector-index \
    --log-driver none \
    code-vector-search:latest
else
  exec docker run --rm -i \
    -v "${PROJECT_ROOT}:/project:ro" \
    -v "${PERSIST_DIR}:/vector-index:rw" \
    -e PROJECT_ROOT=/project \
    -e PERSIST_DIR=/vector-index \
    --log-driver none \
    code-vector-search:latest 2>/dev/null
fi
