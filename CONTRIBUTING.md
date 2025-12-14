# Contributing

Thanks for your interest in improving Code Vector Search! This guide explains how to set up a dev environment and submit changes.

## Getting Started
1. Install Python 3.11+ and Docker (optional, for parity with production usage).
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   The OpenVINO/optimum packages pull native components; installation may take a few minutes.

## Development Workflow
- Run the server locally against a project:
  ```bash
  PROJECT_ROOT=/path/to/code PERSIST_DIR=.vector-index python server.py
  ```
- Dockerized workflow:
  ```bash
  docker build -t code-vector-search .
  PROJECT_ROOT=/path/to/code ./run-docker.sh
  ```
- Tests:
  ```bash
  python test_embedding.py
  ```
  Please ensure tests pass before submitting.

## Coding Standards
- Follow existing patterns in `server.py`; prefer type hints and small, readable functions.
- Keep logs and user-facing text concise; send operational logs to stderr.
- Avoid committing build artifacts or index data; `.vector-index/` is already ignored.

## Pull Requests
- Open a draft PR early if you want feedback.
- Describe the change, rationale, and any trade-offs.
- Note testing performed and any known gaps.
- Keep commits scoped and descriptive; prefer small, reviewable changes.

## Reporting Issues
- Include reproducible steps, expected vs. actual behavior, and environment details (Python version, OS, whether Docker was used).
- For potential security issues, follow the guidance in `SECURITY.md` (do not open a public issue for vulnerabilities).
