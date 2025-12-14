# Code Vector Search (MCP Server)

A Model Context Protocol (MCP) server that builds a local semantic code index using ChromaDB and sentence-transformer embeddings accelerated with Intel OpenVINO. It watches your project, chunks source files, respects `.gitignore`, and exposes semantic search tools to MCP clients.

## Features
- ChromaDB-backed vector store with persistent storage and automatic bloat mitigation.
- Sentence-transformer embeddings with OpenVINO acceleration for Intel CPUs/GPUs (falls back to PyTorch).
- Gitignore-aware indexing with file watching for live updates.
- Chunked documents for precise results with file/line metadata.
- Tools exposed to MCP clients: `semantic_search`, `get_index_stats`, `reindex_project`.

## Quickstart (Docker)
```bash
docker build -t code-vector-search .
PROJECT_ROOT=/path/to/your/code ./run-docker.sh
```
- Project is mounted read-only at `/project`; the vector index persists to `/path/to/your/code/.vector-index`.
- Set `DEBUG=1` to surface server stderr logs from the container.

## Running without Docker
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PROJECT_ROOT=/path/to/your/code PERSIST_DIR=.vector-index python server.py
```

## Configuration
- `PROJECT_ROOT` (default `/project`): directory to index.
- `PERSIST_DIR` (default `/vector-index`): where ChromaDB stores the index.
- `FORCE_REINDEX` (true/false): force a clean rebuild on startup.
- `DEBUG` (with `run-docker.sh`): surface stderr logs instead of silencing them.

## Tests
Run embedding and Chroma integration checks:
```bash
python test_embedding.py
```

## Contributing
See `CONTRIBUTING.md` for setup, style, and pull request guidelines. A `CODE_OF_CONDUCT.md` applies to all community spaces.

## Security
Please review `SECURITY.md` for how to report vulnerabilities.

## License
Licensed under the MIT License. See `LICENSE` for details.
