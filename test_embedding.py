#!/usr/bin/env python3
"""Test the OpenVINO embedding function with ChromaDB."""

import sys
import tempfile
import shutil

def test_embedding_function():
    """Test that OpenVINOEmbeddingFunction works with ChromaDB."""
    print("=" * 60)
    print("Testing OpenVINOEmbeddingFunction")
    print("=" * 60)

    # Import the embedding function from server.py
    from server import OpenVINOEmbeddingFunction

    print("\n1. Creating embedding function...")
    ef = OpenVINOEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    print(f"   Backend: {ef.backend}")

    print("\n2. Testing name() method...")
    name = ef.name()
    print(f"   Name: {name}")
    assert name == "sentence_transformer", f"Expected 'sentence_transformer', got '{name}'"
    print("   ✓ name() works")

    print("\n3. Testing __call__() with list of strings...")
    docs = ["Hello world", "This is a test"]
    result = ef(docs)
    print(f"   Input: {docs}")
    print(f"   Output type: {type(result)}")
    print(f"   Output length: {len(result)}")
    print(f"   First embedding length: {len(result[0])}")
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 2, f"Expected 2 embeddings, got {len(result)}"
    assert isinstance(result[0], list), "Each embedding should be a list"
    assert len(result[0]) == 384, f"Expected 384 dimensions, got {len(result[0])}"
    print("   ✓ __call__() works")

    print("\n4. Testing embed_query() method...")
    query = "search query"
    result = ef.embed_query(query)
    print(f"   Input: {query}")
    print(f"   Output type: {type(result)}")
    print(f"   Output length: {len(result)}")
    print(f"   Inner length: {len(result[0])}")
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 1, f"Expected 1 embedding (wrapped), got {len(result)}"
    assert isinstance(result[0], list), f"Inner should be a list, got {type(result[0])}"
    assert len(result[0]) == 384, f"Expected 384 dimensions, got {len(result[0])}"
    print("   ✓ embed_query() works")

    print("\n5. Testing embed_documents() method...")
    docs = ["doc one", "doc two", "doc three"]
    result = ef.embed_documents(docs)
    print(f"   Input: {docs}")
    print(f"   Output type: {type(result)}")
    print(f"   Output length: {len(result)}")
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 3, f"Expected 3 embeddings, got {len(result)}"
    print("   ✓ embed_documents() works")

    print("\n" + "=" * 60)
    print("All embedding function tests passed!")
    print("=" * 60)


def test_chromadb_integration():
    """Test that the embedding function works with ChromaDB."""
    print("\n" + "=" * 60)
    print("Testing ChromaDB Integration")
    print("=" * 60)

    import chromadb
    from server import OpenVINOEmbeddingFunction

    # Create a temporary directory for the test
    temp_dir = tempfile.mkdtemp()

    try:
        print("\n1. Creating ChromaDB client...")
        client = chromadb.PersistentClient(path=temp_dir)
        print("   ✓ Client created")

        print("\n2. Creating embedding function...")
        ef = OpenVINOEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        print(f"   ✓ Embedding function created (backend: {ef.backend})")

        print("\n3. Creating collection...")
        collection = client.get_or_create_collection(
            name="test_collection",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )
        print("   ✓ Collection created")

        print("\n4. Adding documents...")
        collection.add(
            ids=["doc1", "doc2", "doc3"],
            documents=[
                "Python is a programming language",
                "JavaScript runs in the browser",
                "Docker containers are lightweight"
            ],
            metadatas=[
                {"type": "python"},
                {"type": "javascript"},
                {"type": "docker"}
            ]
        )
        print(f"   ✓ Added 3 documents, total count: {collection.count()}")

        print("\n5. Querying collection...")
        results = collection.query(
            query_texts=["programming languages"],
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )
        print(f"   Query: 'programming languages'")
        print(f"   Results IDs: {results['ids']}")
        print(f"   Results docs: {results['documents']}")
        assert results['ids'], "Should have results"
        assert len(results['ids'][0]) == 2, "Should have 2 results"
        print("   ✓ Query returned results")

        print("\n6. Testing query with single result...")
        results = collection.query(
            query_texts=["containerization"],
            n_results=1,
            include=["documents", "distances"]
        )
        print(f"   Query: 'containerization'")
        print(f"   Top result: {results['documents'][0][0]}")
        assert "Docker" in results['documents'][0][0], "Should find Docker document"
        print("   ✓ Single query works")

        print("\n7. Re-opening collection (persistence test)...")
        # Close and reopen
        del collection
        del client

        client = chromadb.PersistentClient(path=temp_dir)
        ef2 = OpenVINOEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        collection = client.get_or_create_collection(
            name="test_collection",
            embedding_function=ef2,
            metadata={"hnsw:space": "cosine"}
        )
        assert collection.count() == 3, f"Expected 3 documents, got {collection.count()}"
        print(f"   ✓ Collection reopened, count: {collection.count()}")

        print("\n8. Query after reopen...")
        results = collection.query(
            query_texts=["web development"],
            n_results=1
        )
        print(f"   Query: 'web development'")
        print(f"   Top result ID: {results['ids'][0][0]}")
        print("   ✓ Query after reopen works")

        print("\n" + "=" * 60)
        print("All ChromaDB integration tests passed!")
        print("=" * 60)

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        test_embedding_function()
        test_chromadb_integration()
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
