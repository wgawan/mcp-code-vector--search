#!/usr/bin/env python3
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import fnmatch
import hashlib
import sys
import signal
import threading
from contextlib import contextmanager

def _read_positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default

# Intel OpenVINO acceleration support
class OpenVINOEmbeddingFunction:
    """Custom embedding function that uses Intel OpenVINO for acceleration.

    OpenVINO provides optimized inference on Intel CPUs (including older generations)
    and Intel integrated GPUs through hardware-specific optimizations.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self.model = None
        self._model_name = model_name
        self.backend = "pytorch"  # default fallback
        self.batch_size = max(1, int(batch_size))

        # Try to use OpenVINO backend for sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer

            # Try OpenVINO backend first
            try:
                self.model = SentenceTransformer(
                    model_name,
                    backend="openvino",
                    model_kwargs={"device": "CPU"}  # OpenVINO CPU is well-optimized
                )
                self.backend = "openvino"
                print(f"Using OpenVINO backend for Intel-optimized inference", file=sys.stderr)
            except Exception as e:
                print(f"OpenVINO backend not available ({e}), using PyTorch", file=sys.stderr)
                self.model = SentenceTransformer(model_name)
                self.backend = "pytorch"

        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            raise

        print(f"Embedding model loaded with backend: {self.backend}", file=sys.stderr)

    def name(self) -> str:
        """Return the name of the embedding function (required by ChromaDB)."""
        # Use same name as default SentenceTransformer to avoid conflicts with existing indices
        return "sentence_transformer"

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for the input texts (used for documents)."""
        if input is None or len(input) == 0:
            print(f"Warning: __call__ received empty input", file=sys.stderr)
            return [[]]
        try:
            embeddings = self.model.encode(
                input,
                convert_to_numpy=True,
                batch_size=self.batch_size
            )
            result = embeddings.tolist()
            # Ensure we always return List[List[float]]
            if result and not isinstance(result[0], list):
                result = [result]
            return result
        except Exception as e:
            print(f"Embedding error: {e}", file=sys.stderr)
            raise

    def embed_query(self, query: str = None, *, input = None) -> List[List[float]]:
        """Embed a single query string (required by ChromaDB for queries).

        ChromaDB calls this as embed_query(input=...) so we accept both forms.
        Input can be a string or a list of strings.
        Returns List[List[float]] as ChromaDB expects for query_embeddings.
        """
        q = query if query is not None else input
        if not q:
            return [[]]
        # ChromaDB may pass a list with one item
        if isinstance(q, list):
            q = q[0] if q else ""
        if not q:
            return [[]]
        embeddings = self.model.encode(
            q,
            convert_to_numpy=True,
            batch_size=self.batch_size
        )
        # Return as List[List[float]] - wrap single embedding in a list
        return [embeddings.tolist()]

    def embed_documents(self, documents: List[str] = None, *, input: List[str] = None) -> List[List[float]]:
        """Embed a list of documents (required by ChromaDB).

        ChromaDB may call this as embed_documents(input=...) so we accept both forms.
        """
        docs = documents if documents is not None else input
        if not docs:
            return []
        return self.__call__(docs)

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
except ImportError:
    print("ERROR: MCP not installed. Please rebuild Docker image.", file=sys.stderr)
    sys.exit(1)

class GitignoreParser:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.patterns: List[tuple] = []
        self._load_gitignore()
    
    def _load_gitignore(self):
        """Load and parse .gitignore file"""
        gitignore_path = self.project_root / '.gitignore'
        
        if not gitignore_path.exists():
            print("No .gitignore found, using default ignores only", file=sys.stderr)
            return
        
        try:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    is_negation = line.startswith('!')
                    if is_negation:
                        line = line[1:]
                    
                    self.patterns.append((line, is_negation, self.project_root))
            
            print(f"Loaded {len(self.patterns)} patterns from .gitignore", file=sys.stderr)
        except Exception as e:
            print(f"Error reading .gitignore: {e}", file=sys.stderr)
    
    def _matches_pattern(self, filepath: Path, pattern: str, base_dir: Path) -> bool:
        """Check if filepath matches a gitignore pattern"""
        try:
            rel_path = filepath.relative_to(base_dir)
        except ValueError:
            return False
        
        rel_path_str = str(rel_path)
        
        if pattern.endswith('/'):
            pattern = pattern.rstrip('/')
            for parent in rel_path.parents:
                if fnmatch.fnmatch(parent.name, pattern):
                    return True
            return False
        
        if '/' in pattern:
            return fnmatch.fnmatch(rel_path_str, pattern)
        
        if fnmatch.fnmatch(filepath.name, pattern):
            return True
        
        for parent in rel_path.parents:
            if parent == Path('.'):
                break
            if fnmatch.fnmatch(parent.name, pattern):
                return True
        
        return False
    
    def should_ignore(self, filepath: str) -> bool:
        """Check if file should be ignored based on .gitignore rules"""
        filepath = Path(filepath)
        should_ignore = False
        
        for pattern, is_negation, base_dir in self.patterns:
            matches = self._matches_pattern(filepath, pattern, base_dir)
            if matches:
                should_ignore = not is_negation
        
        return should_ignore


class CodeVectorStore:
    def __init__(self, project_root: str, persist_directory: Optional[str] = None):
        self.project_root = project_root
        self.gitignore_parser = GitignoreParser(project_root)
        self.chunk_size = _read_positive_int_env("CHUNK_SIZE", 800)
        self.index_batch_size = _read_positive_int_env("INDEX_BATCH_CHUNKS", 512)
        self.embedding_batch_size = _read_positive_int_env("EMBED_BATCH_SIZE", 64)
        self.embedding_model_name = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
        self.always_ignore_dirs = {'.git', '__pycache__', '.pytest_cache',
                                   '.mypy_cache', '.tox', 'venv', '.venv',
                                   'node_modules', '.next', 'dist', 'build',
                                   'target', '.gradle', '.idea', '.vscode'}

        # Lock to prevent race conditions between file watcher and reset operations
        self._collection_lock = threading.RLock()

        # Initialize Chroma client
        if persist_directory:
            print(f"Using persistent storage at: {persist_directory}", file=sys.stderr)
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            print("Using in-memory storage (will not persist)", file=sys.stderr)
            self.client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))
        
        # Use sentence transformers for embeddings with Intel OpenVINO acceleration
        self.embedding_function = OpenVINOEmbeddingFunction(
            model_name=self.embedding_model_name,
            batch_size=self.embedding_batch_size
        )
        print(
            f"Index settings: CHUNK_SIZE={self.chunk_size}, "
            f"INDEX_BATCH_CHUNKS={self.index_batch_size}, "
            f"EMBED_BATCH_SIZE={self.embedding_batch_size}, "
            f"BACKEND={self.embedding_function.backend}, "
            f"MODEL={self.embedding_function._model_name}",
            file=sys.stderr
        )
        print(
            "Model options: all-MiniLM-L12-v2 (higher quality, slower), "
            "multi-qa-MiniLM-L6-cos-v1 (QA-style queries), "
            "bge-small-en-v1.5 (faster, smaller).",
            file=sys.stderr
        )
        
        # Create collection name based on project root hash
        collection_name = f"code_{hashlib.md5(project_root.encode()).hexdigest()[:8]}"
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"Vector store initialized. Collection: {collection_name}, Chunks: {self.collection.count()}", file=sys.stderr)

    def reset_collection(self):
        """Delete and recreate collection to reclaim HNSW disk space.

        Chroma's HNSW index uses append-only storage - deletions mark records
        as deleted but don't reclaim space. This method forces fresh files.
        """
        with self._collection_lock:
            collection_name = self.collection.name
            print(f"🔄 Resetting collection {collection_name} to reclaim disk space...", file=sys.stderr)

            # Delete the collection entirely (removes HNSW files)
            self.client.delete_collection(name=collection_name)

            # Recreate with same settings
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✓ Collection reset complete", file=sys.stderr)

    def _compute_hash(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()
    
    def _should_index_file(self, filepath: str) -> bool:
        """Filter files to index"""
        extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', 
                     '.c', '.h', '.go', '.rs', '.rb', '.php', '.swift', '.kt',
                     '.scala', '.css', '.scss', '.less', '.html', '.vue',
                     '.md', '.json', '.yaml', '.yml', '.toml', '.ini',
                     '.sh', '.bash', '.sql', '.xml', '.txt', '.conf'}
        
        path = Path(filepath)
        
        if any(ignored in path.parts for ignored in self.always_ignore_dirs):
            return False
        
        if self.gitignore_parser.should_ignore(filepath):
            return False
        
        if path.suffix not in extensions:
            return False
        
        try:
            if os.path.getsize(filepath) > 1_000_000:
                print(f"Skipping large file: {filepath}", file=sys.stderr)
                return False
        except OSError:
            return False
        
        return True
    
    def _chunk_file(self, content: str, filepath: str, chunk_size: int = 500) -> List[Dict]:
        """Split file into chunks for better granularity"""
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            current_size += len(line)
            
            if current_size >= chunk_size or i == len(lines) - 1:
                chunk_text = '\n'.join(current_chunk)
                if chunk_text.strip():
                    chunks.append({
                        'text': chunk_text,
                        'filepath': filepath,
                        'start_line': i - len(current_chunk) + 2,
                        'end_line': i + 1
                    })
                current_chunk = []
                current_size = 0
        
        return chunks

    def _add_batch(self, ids: List[str], documents: List[str], metadatas: List[Dict]):
        if not ids:
            return
        print(f"  Adding {len(ids)} chunks...", file=sys.stderr, end='', flush=True)
        with self._collection_lock:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        print(f" done", file=sys.stderr, flush=True)
        ids.clear()
        documents.clear()
        metadatas.clear()
    
    def index_file(self, filepath: str):
        """Index a single file using Chroma"""
        if not self._should_index_file(filepath):
            return

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            try:
                file_stat = os.stat(filepath)
                file_mtime = int(file_stat.st_mtime)
                file_size = file_stat.st_size
            except OSError:
                file_mtime = None
                file_size = None

            content_hash = self._compute_hash(content)

            # Check if file already indexed with same hash (lock for collection access)
            with self._collection_lock:
                try:
                    existing = self.collection.get(
                        where={"filepath": filepath},
                        include=["metadatas"]
                    )

                    if existing['ids'] and existing['metadatas']:
                        if existing['metadatas'][0].get('content_hash') == content_hash:
                            return  # No changes
                except:
                    pass

            # Remove old chunks for this file (has its own locking)
            self.remove_file(filepath)

            # Create chunks (no lock needed - pure computation)
            chunks = self._chunk_file(content, filepath, chunk_size=self.chunk_size)

            if not chunks:
                return

            # Prepare data for Chroma (no lock needed - pure computation)
            ids = [f"{filepath}::{i}" for i in range(len(chunks))]
            documents = [chunk['text'] for chunk in chunks]
            metadatas = [
                {
                    'filepath': chunk['filepath'],
                    'start_line': chunk['start_line'],
                    'end_line': chunk['end_line'],
                    'content_hash': content_hash,
                    'file_mtime': file_mtime,
                    'file_size': file_size,
                    'chunk_index': i
                }
                for i, chunk in enumerate(chunks)
            ]

            # Add to Chroma (lock for collection access)
            print(f"  Adding {len(chunks)} chunks...", file=sys.stderr, end='', flush=True)
            sys.stderr.flush()

            with self._collection_lock:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )

            print(f" done", file=sys.stderr, flush=True)
            sys.stderr.flush()
            print(f"✓ Indexed: {filepath} ({len(chunks)} chunks)", file=sys.stderr, flush=True)

        except Exception as e:
            print(f"✗ Error indexing {filepath}: {e}", file=sys.stderr)
    
    def remove_file(self, filepath: str):
        """Remove file from index"""
        try:
            with self._collection_lock:
                results = self.collection.get(
                    where={"filepath": filepath},
                    include=[]
                )

                if results['ids']:
                    self.collection.delete(ids=results['ids'])
                    print(f"✓ Removed: {filepath} ({len(results['ids'])} chunks)", file=sys.stderr)
        except Exception as e:
            pass
    
    def index_directory(self, directory: str, *, fresh_index: bool = False):
        """Index entire directory"""
        if fresh_index:
            return self._index_directory_fresh(directory)

        print(f"\n📁 Indexing directory: {directory}", file=sys.stderr)
        indexed_count = 0
        skipped_count = 0
        
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not self.gitignore_parser.should_ignore(os.path.join(root, d))
                      and d not in self.always_ignore_dirs]
            
            for file in files:
                filepath = os.path.join(root, file)
                if self._should_index_file(filepath):
                    self.index_file(filepath)
                    indexed_count += 1
                else:
                    skipped_count += 1
        
        print(f"\n✅ Indexing complete!", file=sys.stderr)
        print(f"   Indexed: {indexed_count} files", file=sys.stderr)
        print(f"   Skipped: {skipped_count} files", file=sys.stderr)
        with self._collection_lock:
            print(f"   Total chunks: {self.collection.count()}\n", file=sys.stderr)

    def _index_directory_fresh(self, directory: str):
        print(f"\n📁 Indexing directory: {directory}", file=sys.stderr)
        print(f"   Fast path enabled (chunk size: {self.chunk_size}, batch: {self.index_batch_size})", file=sys.stderr)
        indexed_count = 0
        skipped_count = 0
        batch_ids: List[str] = []
        batch_docs: List[str] = []
        batch_metas: List[Dict] = []

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not self.gitignore_parser.should_ignore(os.path.join(root, d))
                      and d not in self.always_ignore_dirs]

            for file in files:
                filepath = os.path.join(root, file)
                if not self._should_index_file(filepath):
                    skipped_count += 1
                    continue

                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    file_stat = os.stat(filepath)
                    file_mtime = int(file_stat.st_mtime)
                    file_size = file_stat.st_size
                except Exception:
                    skipped_count += 1
                    continue

                content_hash = self._compute_hash(content)
                chunks = self._chunk_file(content, filepath, chunk_size=self.chunk_size)
                if not chunks:
                    skipped_count += 1
                    continue

                for i, chunk in enumerate(chunks):
                    batch_ids.append(f"{filepath}::{i}")
                    batch_docs.append(chunk['text'])
                    batch_metas.append({
                        'filepath': chunk['filepath'],
                        'start_line': chunk['start_line'],
                        'end_line': chunk['end_line'],
                        'content_hash': content_hash,
                        'file_mtime': file_mtime,
                        'file_size': file_size,
                        'chunk_index': i
                    })

                indexed_count += 1
                if len(batch_ids) >= self.index_batch_size:
                    self._add_batch(batch_ids, batch_docs, batch_metas)
                print(f"✓ Indexed: {filepath} ({len(chunks)} chunks)", file=sys.stderr, flush=True)

        self._add_batch(batch_ids, batch_docs, batch_metas)

        print(f"\n✅ Indexing complete!", file=sys.stderr)
        print(f"   Indexed: {indexed_count} files", file=sys.stderr)
        print(f"   Skipped: {skipped_count} files", file=sys.stderr)
        with self._collection_lock:
            print(f"   Total chunks: {self.collection.count()}\n", file=sys.stderr)
    
    def search(self, query: str, top_k: int = 5, filter_filepath: Optional[str] = None) -> List[Dict]:
        """Search for relevant code chunks"""
        with self._collection_lock:
            if self.collection.count() == 0:
                return []

            where_filter = None
            if filter_filepath:
                where_filter = {"filepath": {"$eq": filter_filepath}}

            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=min(top_k, self.collection.count()),
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                print(f"Search error: {e}", file=sys.stderr)
                raise

            if not results['ids'] or not results['ids'][0]:
                return []

            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'filepath': results['metadatas'][0][i]['filepath'],
                    'start_line': results['metadatas'][0][i]['start_line'],
                    'end_line': results['metadatas'][0][i]['end_line'],
                    'score': results['distances'][0][i]
                })

            return formatted_results
    
    def get_stats(self) -> Dict:
        """Get statistics about indexed code"""
        with self._collection_lock:
            all_items = self.collection.get(include=["metadatas"])

            files = set()
            for metadata in all_items['metadatas']:
                files.add(metadata['filepath'])

            return {
                'total_chunks': self.collection.count(),
                'total_files': len(files),
                'unique_files': sorted(list(files))
            }

    def get_persist_dir_size_mb(self, persist_dir: str) -> float:
        """Get total size of persistence directory in MB"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(persist_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_size += os.path.getsize(fp)
                except OSError:
                    pass
        return total_size / (1024 * 1024)

    def sync_with_filesystem(self, directory: str) -> Dict:
        """Sync index with actual filesystem state.

        - Removes indexed files that no longer exist
        - Re-indexes files that have changed (based on content hash)
        - Indexes new files

        If >20% of files need changes, performs a clean reset to avoid HNSW bloat.

        Returns stats about what was done.
        """
        stats = {'removed': 0, 'updated': 0, 'added': 0, 'unchanged': 0, 'reset': False}

        # Get all indexed files and their hashes (lock for collection access)
        with self._collection_lock:
            all_items = self.collection.get(include=["metadatas"])
        indexed_files: Dict[str, Dict[str, Any]] = {}  # filepath -> metadata
        for metadata in all_items['metadatas']:
            filepath = metadata['filepath']
            if filepath not in indexed_files:
                indexed_files[filepath] = {
                    'content_hash': metadata.get('content_hash', ''),
                    'file_mtime': metadata.get('file_mtime'),
                    'file_size': metadata.get('file_size'),
                }

        # First pass: collect changes needed
        files_to_add = []  # (filepath,)
        files_to_update = []  # (filepath,)
        files_to_remove = []  # filepaths that no longer exist
        existing_indexed_files = set()

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not self.gitignore_parser.should_ignore(os.path.join(root, d))
                      and d not in self.always_ignore_dirs]

            for file in files:
                filepath = os.path.join(root, file)
                if not self._should_index_file(filepath):
                    continue

                if filepath not in indexed_files:
                    files_to_add.append(filepath)
                    continue

                existing_indexed_files.add(filepath)
                entry = indexed_files[filepath]

                try:
                    file_stat = os.stat(filepath)
                    current_mtime = int(file_stat.st_mtime)
                    current_size = file_stat.st_size
                except OSError:
                    continue

                stored_mtime = entry.get('file_mtime')
                stored_size = entry.get('file_size')
                stored_hash = entry.get('content_hash')

                if stored_mtime is not None and stored_size is not None:
                    if stored_mtime == current_mtime and stored_size == current_size:
                        continue

                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    current_hash = self._compute_hash(content)
                except Exception:
                    continue

                if stored_hash != current_hash:
                    files_to_update.append(filepath)

        # Find deleted files
        for filepath in indexed_files:
            if filepath not in existing_indexed_files:
                files_to_remove.append(filepath)

        total_indexed = len(indexed_files)
        total_changes = len(files_to_add) + len(files_to_update) + len(files_to_remove)

        # If many changes needed, do clean reset to avoid HNSW bloat
        if total_indexed > 0 and total_changes > 0:
            change_ratio = total_changes / total_indexed
            if change_ratio > 0.2 or total_changes > 100:
                print(f"   Many changes detected ({total_changes}), doing clean reset...", file=sys.stderr)
                self.reset_collection()
                self.index_directory(directory, fresh_index=True)
                stats['reset'] = True
                stats['added'] = len(files_to_add) + len(indexed_files) - len(files_to_remove)
                stats['updated'] = len(files_to_update)
                stats['removed'] = len(files_to_remove)
                return stats

        # Incremental sync for small changes
        for filepath in files_to_remove:
            self.remove_file(filepath)
            stats['removed'] += 1

        for filepath in files_to_update:
            self.remove_file(filepath)
            self.index_file(filepath)
            stats['updated'] += 1

        for filepath in files_to_add:
            self.index_file(filepath)
            stats['added'] += 1

        stats['unchanged'] = len(existing_indexed_files) - len(files_to_update)

        return stats


class CodeIndexHandler(FileSystemEventHandler):
    def __init__(self, vector_store: CodeVectorStore, project_root: str):
        self.vector_store = vector_store
        self.project_root = project_root
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path != os.path.join(self.project_root, '.gitignore'):
            self.vector_store.index_file(event.src_path)
        elif event.src_path == os.path.join(self.project_root, '.gitignore'):
            print("📝 .gitignore modified, reloading...", file=sys.stderr)
            self.vector_store.gitignore_parser._load_gitignore()
    
    def on_created(self, event):
        if not event.is_directory:
            self.vector_store.index_file(event.src_path)
    
    def on_deleted(self, event):
        if not event.is_directory:
            self.vector_store.remove_file(event.src_path)
    
    def on_moved(self, event):
        if not event.is_directory:
            self.vector_store.remove_file(event.src_path)
            self.vector_store.index_file(event.dest_path)


# MCP Server Implementation
app = Server("code-vector-search")
vector_store = None
observer = None

@app.list_tools()
async def list_tools() -> List[Tool]:
    return [
        Tool(
            name="semantic_search",
            description="Fast semantic code search. Use this first to locate relevant files or modules before listing directories or grepping. Ideal for understanding where a feature lives; follow up with file inspection after finding hits"
                       "Semantic search over the codebase that works even without exact keywords and respects .gitignore rules.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g., 'authentication logic', 'for loops with await inside'). Attempt using this tool before trying to grep"
                    },
                    "top_k": {
                        "type": "number",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    },
                    "filter_filepath": {
                        "type": "string",
                        "description": "Optional: Only search within a specific file"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_index_stats",
            description="Get statistics about the indexed codebase (number of files, chunks, etc.)",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="reindex_project",
            description="Force a full re-index of the project directory. Use if index seems out of sync.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    global vector_store

    if vector_store is None:
        return [TextContent(
            type="text",
            text="Error: Vector store not initialized. Please check server logs for initialization errors."
        )]

    try:
        if name == "semantic_search":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            filter_filepath = arguments.get("filter_filepath")

            results = vector_store.search(query, top_k, filter_filepath)

            if not results:
                return [TextContent(
                    type="text",
                    text="No results found. The codebase may not be indexed yet or the query didn't match any code."
                )]

            response = f"Found {len(results)} relevant code chunks:\n\n"
            for i, result in enumerate(results, 1):
                response += f"**Result {i}** (relevance score: {result['score']:.4f})\n"
                response += f"File: `{result['filepath']}`\n"
                response += f"Lines: {result['start_line']}-{result['end_line']}\n\n"
                response += f"```\n{result['text']}\n```\n\n"

            return [TextContent(type="text", text=response)]

        elif name == "get_index_stats":
            stats = vector_store.get_stats()
            response = f"**Index Statistics**\n\n"
            response += f"📊 Total chunks: {stats['total_chunks']}\n"
            response += f"📁 Total files: {stats['total_files']}\n\n"

            if stats['unique_files']:
                response += f"**Indexed files (showing first 50):**\n"
                for filepath in stats['unique_files'][:50]:
                    response += f"- {filepath}\n"
                if len(stats['unique_files']) > 50:
                    response += f"\n... and {len(stats['unique_files']) - 50} more files\n"
            else:
                response += "No files indexed yet.\n"

            return [TextContent(type="text", text=response)]

        elif name == "reindex_project":
            project_root = os.environ.get("PROJECT_ROOT", ".")

            # Reset collection to reclaim disk space (HNSW is append-only)
            vector_store.reset_collection()

            vector_store.index_directory(project_root, fresh_index=True)

            return [TextContent(
                type="text",
                text=f"✅ Re-indexed project at {project_root}\n"
                     f"Total chunks: {vector_store.collection.count()}"
            )]

        else:
            return [TextContent(
                type="text",
                text=f"Error: Unknown tool '{name}'"
            )]

    except Exception as e:
        print(f"Error in call_tool({name}): {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return [TextContent(
            type="text",
            text=f"Error executing tool '{name}': {str(e)}"
        )]


_cleanup_done = False

def cleanup():
    """Clean shutdown - stop observer and ensure DB is flushed"""
    global vector_store, observer, _cleanup_done

    if _cleanup_done:
        return
    _cleanup_done = True

    print("\n🛑 Shutting down gracefully...", file=sys.stderr)

    if observer:
        print("   Stopping file watcher...", file=sys.stderr)
        observer.stop()
        observer.join(timeout=5)

    # ChromaDB's PersistentClient should auto-flush, but we can help by
    # ensuring any pending operations complete
    if vector_store and hasattr(vector_store, 'client'):
        print("   Flushing database...", file=sys.stderr)
        # Force a sync by doing a count operation
        try:
            vector_store.collection.count()
        except:
            pass

    print("✓ Shutdown complete", file=sys.stderr)


async def main():
    global vector_store, observer

    project_root = os.environ.get("PROJECT_ROOT", "/project")
    persist_dir = os.environ.get("PERSIST_DIR", "/vector-index")
    force_reindex = os.environ.get("FORCE_REINDEX", "").lower() in ("1", "true", "yes")

    if not os.path.exists(project_root):
        print(f"ERROR: Project root does not exist: {project_root}", file=sys.stderr)
        return

    print(f"🚀 Initializing Code Vector Search", file=sys.stderr)
    print(f"   Project: {project_root}", file=sys.stderr)
    print(f"   Python: {sys.version}", file=sys.stderr)

    vector_store = CodeVectorStore(project_root, persist_directory=persist_dir)

    chunk_count = vector_store.collection.count()

    if force_reindex:
        print(f"🔄 FORCE_REINDEX=true, performing clean reindex...", file=sys.stderr)
        vector_store.reset_collection()
        vector_store.index_directory(project_root, fresh_index=True)
    elif chunk_count == 0:
        # Fresh index
        vector_store.index_directory(project_root, fresh_index=True)
    else:
        # Check for index bloat (HNSW append-only issue)
        persist_size_mb = vector_store.get_persist_dir_size_mb(persist_dir)
        # HNSW indices with 384-dim embeddings need ~15KB per chunk
        # Only flag as bloat if significantly over that (e.g., 3x or more)
        expected_size_mb = chunk_count * 0.015  # ~15KB per chunk estimate
        bloat_ratio = persist_size_mb / max(expected_size_mb, 1)

        print(f"📚 Loaded existing index with {chunk_count} chunks", file=sys.stderr)
        print(f"   Index size: {persist_size_mb:.1f} MB (expected ~{expected_size_mb:.1f} MB)", file=sys.stderr)

        if bloat_ratio > 5:
            print(f"⚠️  Index bloat detected ({bloat_ratio:.1f}x expected size)!", file=sys.stderr)
            print(f"   Performing clean reindex to reclaim space...", file=sys.stderr)
            vector_store.reset_collection()
            vector_store.index_directory(project_root, fresh_index=True)
        else:
            # Sync with filesystem to catch changes without bloating
            print(f"🔄 Syncing index with filesystem...", file=sys.stderr)
            sync_stats = vector_store.sync_with_filesystem(project_root)
            if sync_stats.get('reset'):
                print(f"   Full reset performed due to many changes", file=sys.stderr)
            print(f"   Sync complete: {sync_stats['added']} added, {sync_stats['updated']} updated, "
                  f"{sync_stats['removed']} removed, {sync_stats['unchanged']} unchanged", file=sys.stderr)
            print(f"   Total chunks: {vector_store.collection.count()}", file=sys.stderr)
    
    # Set up file watcher
    event_handler = CodeIndexHandler(vector_store, project_root)
    observer = Observer()
    observer.schedule(event_handler, project_root, recursive=True)
    observer.start()
    print(f"👀 File watcher started", file=sys.stderr)
    print(f"💡 Send SIGTERM or SIGINT (Ctrl+C) for graceful shutdown\n", file=sys.stderr)

    # Set up signal handlers for graceful shutdown
    def handle_signal(signum, frame):
        print(f"\n📡 Received signal {signum}", file=sys.stderr)
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    except Exception as e:
        # Handle disconnection or stream errors gracefully
        error_msg = str(e)
        if "EOF" in error_msg or "validation error" in error_msg.lower():
            print(f"Client disconnected", file=sys.stderr)
        else:
            print(f"Server error: {e}", file=sys.stderr)
    finally:
        cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
