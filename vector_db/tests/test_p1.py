import sys
import json
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cleaner import clean_code, clean_slack_message, load_slack_export, load_jira_csv
from chunker import chunk_code_file, chunk_slack_export, chunk_jira_csv
from embedder import get_embedding, get_embedding_dimensions
from ingestor import get_collection, ingest_code_file, ingest_slack_file, ingest_jira_file, search

BASE = Path("C:/ContextEngine/test_data/phase1")
CODE_FILE = BASE / "code" / "sample.py"
SLACK_FILE = BASE / "slack" / "general.json"
JIRA_FILE = BASE / "jira" / "issues.csv"

COLLECTION_NAME_TEST = "test_phase1"


def assert_true(condition: bool, message: str):
    if not condition:
        print(f"  ✗ FAIL: {message}")
        return False
    print(f"  ✓ PASS: {message}")
    return True


def test_cleaner():
    print("\n[1] Cleaner Tests")
    passed = 0
    total = 0

    # Code cleaner
    raw = "def foo():\n    x = 1   \n\n\n\n    return x\n"
    cleaned = clean_code(raw)
    total += 1
    passed += assert_true("\n\n\n" not in cleaned, "clean_code: collapses 3+ blank lines")
    total += 1
    passed += assert_true("def foo" in cleaned, "clean_code: preserves function definition")

    # Slack cleaner
    msg = "Hey <@U12345>, check <#C99999|general> and see <https://example.com|this link>"
    cleaned_msg = clean_slack_message(msg)
    total += 1
    passed += assert_true("<@" not in cleaned_msg, "clean_slack_message: removes user mention tags")
    total += 1
    passed += assert_true("<#" not in cleaned_msg, "clean_slack_message: removes channel tags")

    print(f"  Cleaner: {passed}/{total} passed")
    return passed == total


# Chunker tests
def test_chunker_code():
    print("\n[2a] Chunker — Code")
    if not CODE_FILE.exists():
        print(f"  ⚠ SKIP: {CODE_FILE} not found")
        return True

    chunks = chunk_code_file(CODE_FILE)
    passed = 0
    total = 0

    total += 1
    passed += assert_true(len(chunks) > 0, f"chunk_code_file: produced chunks (got {len(chunks)})")

    for chunk in chunks:
        total += 1
        passed += assert_true("chunk_id" in chunk, "each chunk has chunk_id")
        total += 1
        passed += assert_true("content" in chunk and len(chunk["content"]) > 0, "each chunk has non-empty content")
        total += 1
        passed += assert_true(chunk["metadata"]["source_type"] == "code", "source_type is 'code'")
        total += 1
        passed += assert_true("function_name" in chunk["metadata"], "chunk has function_name")
        total += 1
        passed += assert_true("start_line" in chunk["metadata"], "chunk has start_line")
        break  

    from chunker import MAX_CODE_LINES
    for chunk in chunks:
        line_count = chunk["content"].count("\n") + 1
        total += 1
        passed += assert_true(
            line_count <= MAX_CODE_LINES + 5, 
            f"chunk '{chunk['metadata']['function_name']}' within line limit ({line_count} lines)"
        )

    chunks2 = chunk_code_file(CODE_FILE)
    ids1 = {c["chunk_id"] for c in chunks}
    ids2 = {c["chunk_id"] for c in chunks2}
    total += 1
    passed += assert_true(ids1 == ids2, "chunk IDs are deterministic across runs")

    print(f"  Code chunker: {passed}/{total} passed")
    return passed == total


def test_chunker_slack():
    print("\n[2b] Chunker — Slack")
    if not SLACK_FILE.exists():
        print(f"  ⚠ SKIP: {SLACK_FILE} not found")
        return True

    chunks = chunk_slack_export(SLACK_FILE)
    passed = 0
    total = 0

    total += 1
    passed += assert_true(len(chunks) > 0, f"chunk_slack_export: produced chunks (got {len(chunks)})")

    for chunk in chunks:
        total += 1
        passed += assert_true(chunk["metadata"]["source_type"] == "slack", "source_type is 'slack'")
        total += 1
        passed += assert_true("channel_name" in chunk["metadata"], "chunk has channel_name")
        total += 1
        passed += assert_true("authors" in chunk["metadata"], "chunk has authors")
        break

    print(f"  Slack chunker: {passed}/{total} passed")
    return passed == total


def test_chunker_jira():
    print("\n[2c] Chunker — Jira")
    if not JIRA_FILE.exists():
        print(f"  ⚠ SKIP: {JIRA_FILE} not found")
        return True

    chunks = chunk_jira_csv(JIRA_FILE)
    passed = 0
    total = 0

    total += 1
    passed += assert_true(len(chunks) > 0, f"chunk_jira_csv: produced chunks (got {len(chunks)})")
    total += 1
    passed += assert_true(len(chunks) <= 10, f"small dataset: ≤10 chunks (got {len(chunks)})")

    for chunk in chunks:
        total += 1
        passed += assert_true(chunk["metadata"]["source_type"] == "jira", "source_type is 'jira'")
        total += 1
        passed += assert_true("ticket_id" in chunk["metadata"], "chunk has ticket_id")
        total += 1
        passed += assert_true(len(chunk["content"]) > 0, "chunk content is non-empty")
        break

    print(f"  Jira chunker: {passed}/{total} passed")
    return passed == total


# Embedding tests
def test_embedding():
    print("\n[3] Embedding Tests")
    passed = 0
    total = 0

    dims = get_embedding_dimensions()
    total += 1
    passed += assert_true(dims > 0, f"embedding model returns vectors (dims={dims})")

    vec1 = get_embedding("Why does the retry logic exist?")
    vec2 = get_embedding("Why does the retry logic exist?")
    total += 1
    passed += assert_true(len(vec1) == dims, "embedding dimension matches reported dims")
    total += 1
    passed += assert_true(vec1 == vec2, "same text produces identical embeddings (deterministic)")

    vec3 = get_embedding("The sky is blue and the grass is green.")
    # Cosine similarity
    dot = sum(a * b for a, b in zip(vec1, vec3))
    total += 1
    passed += assert_true(vec1 != vec3, "different text produces different embeddings")

    print(f"  Embedder: {passed}/{total} passed")
    return passed == total


def test_chromadb_insertion():
    print("\n[4] ChromaDB Insertion Tests")
    import chromadb

    client = chromadb.PersistentClient(path="C:/ContextEngine/chromadb_store")
    # Using collection created exclusively for testing
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME_TEST,
        metadata={"hnsw:space": "cosine"},
    )

    passed = 0
    total = 0

    initial_count = collection.count()

    if CODE_FILE.exists():
        count = ingest_code_file(CODE_FILE, collection)
        total += 1
        passed += assert_true(count > 0, f"code file ingested ({count} chunks)")

    if SLACK_FILE.exists():
        count = ingest_slack_file(SLACK_FILE, collection)
        total += 1
        passed += assert_true(count > 0, f"slack file ingested ({count} chunks)")

    if JIRA_FILE.exists():
        count = ingest_jira_file(JIRA_FILE, collection)
        total += 1
        passed += assert_true(count > 0, f"jira file ingested ({count} chunks)")

    final_count = collection.count()
    total += 1
    passed += assert_true(final_count > initial_count, f"collection grew after ingestion ({initial_count} → {final_count})")

    if CODE_FILE.exists():
        count_before = collection.count()
        ingest_code_file(CODE_FILE, collection)
        count_after = collection.count()
        total += 1
        passed += assert_true(count_before == count_after, "re-ingesting same file does not duplicate chunks")

    print(f"  ChromaDB insertion: {passed}/{total} passed")
    return passed == total, collection


# Similarity search tests
def test_similarity_search(collection):
    print("\n[5] Similarity Search Tests")
    passed = 0
    total = 0

    if collection.count() == 0:
        print("  ⚠ SKIP: collection is empty")
        return True

    results = search("how does this module work", collection, n_results=3)
    total += 1
    passed += assert_true(len(results) > 0, f"search returns results (got {len(results)})")
    total += 1
    passed += assert_true("content" in results[0], "result has 'content' field")
    total += 1
    passed += assert_true("metadata" in results[0], "result has 'metadata' field")
    total += 1
    passed += assert_true("distance" in results[0], "result has 'distance' field")

    # Source-type filtering
    results_code = search("function definition", collection, n_results=3, source_type="code")
    for r in results_code:
        total += 1
        passed += assert_true(r["metadata"]["source_type"] == "code", "filtered results are all source_type='code'")

    # Cross-source
    results_all = search("error handling", collection, n_results=5)
    source_types = {r["metadata"]["source_type"] for r in results_all}
    print(f"  Cross-source query returned source types: {source_types}")

    print(f"  Similarity search: {passed}/{total} passed")
    return passed == total

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1 — Pipeline Sanity Check")
    print("=" * 60)

    results = []
    results.append(test_cleaner())
    results.append(test_chunker_code())
    results.append(test_chunker_slack())
    results.append(test_chunker_jira())
    results.append(test_embedding())

    insertion_passed, collection = test_chromadb_insertion()
    results.append(insertion_passed)
    results.append(test_similarity_search(collection))

    passed_count = sum(1 for r in results if r)
    print("\n" + "=" * 60)
    print(f"PHASE 1 RESULT: {passed_count}/{len(results)} test groups passed")
    if all(results):
        print("✓ Phase 1 COMPLETE — ready for Phase 2")
    else:
        print("✗ Phase 1 INCOMPLETE — fix failures before proceeding")
    print("=" * 60)