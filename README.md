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
- `CHUNK_SIZE` (default `800`): target characters per chunk; larger values reduce chunks and speed indexing.
- `EMBED_BATCH_SIZE` (default `64`): sentence-transformer encode batch size.
- `EMBED_MODEL_NAME` (default `all-MiniLM-L6-v2`): embedding model name for SentenceTransformers.
- `INDEX_BATCH_CHUNKS` (default `512`): number of chunks per Chroma add batch during full indexing.
- `DEBUG` (with `run-docker.sh`): surface stderr logs instead of silencing them.

## Performance tips
- These defaults are tuned for a 4-core, 32GB machine; adjust down if you see memory pressure.
- Increase `EMBED_BATCH_SIZE` (e.g., `96`) if you have enough headroom; lower it if you see OOMs.
- For very large repos, prefer a fresh index after big changes to avoid HNSW bloat.
- Model choices: `all-MiniLM-L12-v2` (higher quality, slower), `multi-qa-MiniLM-L6-cos-v1` (QA-style queries), `bge-small-en-v1.5` (faster, smaller).

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
