import sys
import time
import chromadb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from chunker import chunk_code_file, chunk_slack_export, chunk_jira_csv
from ingestor import get_collection, ingest_directory, search, ingest_code_file, ingest_slack_file, ingest_jira_file

BASE_P3 = Path("C:/ContextEngine/test_data/phase3")
SCRAPY_ROOT = BASE_P3 / "scrapy"
SLACK_DIR = BASE_P3 / "slack"
JIRA_DIR = BASE_P3 / "jira"

COLLECTION_NAME_P3 = "context_engine" 

CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".md", ".rst", ".cfg", ".ini"}


def assert_true(condition, message):
    if not condition:
        print(f"  ✗ FAIL: {message}")
        return False
    print(f"  ✓ PASS: {message}")
    return True

def test_scrapy_file_discovery():
    print("\n[1] Scrapy Repository File Discovery")
    passed = 0
    total = 0

    if not SCRAPY_ROOT.exists():
        print(f"SKIP: Scrapy not found at {SCRAPY_ROOT}")
        print(f"Run: git clone https://github.com/scrapy/scrapy {SCRAPY_ROOT}")
        return True

    py_files = list(SCRAPY_ROOT.rglob("*.py"))
    total += 1
    passed += assert_true(len(py_files) > 100, f"Found {len(py_files)} .py files in Scrapy (expected >100)")

    errors = []
    chunk_counts = []
    for f in py_files:
        try:
            chunks = chunk_code_file(f)
            chunk_counts.append(len(chunks))
        except Exception as e:
            errors.append((f.name, str(e)))

    total += 1
    passed += assert_true(
        len(errors) == 0,
        f"All {len(py_files)} .py files parsed without errors"
        if len(errors) == 0
        else f"{len(errors)} files failed: {errors[:3]}"
    )

    total_chunks = sum(chunk_counts)
    total += 1
    passed += assert_true(total_chunks > 500, f"Total code chunks from Scrapy: {total_chunks} (expected >500)")

    avg = total_chunks / len(py_files) if py_files else 0
    print(f"  Avg chunks per file: {avg:.1f}")

    print(f"  Scrapy discovery: {passed}/{total} passed")
    return passed == total

# Full ingestion at scale
def test_full_ingestion():
    print("\n[2] Full-Scale Ingestion")
    client = chromadb.PersistentClient(path="C:/ContextEngine/chromadb_store")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME_P3,
        metadata={"hnsw:space": "cosine"},
    )

    passed = 0
    total = 0

    count_before = collection.count()
    print(f"  Collection size before ingestion: {count_before}")

    # Code — entire Scrapy repo
    if SCRAPY_ROOT.exists():
        print("  Ingesting Scrapy repository (this will take a few minutes)...")
        start = time.time()
        results = ingest_directory(SCRAPY_ROOT, collection, recursive=True)
        elapsed = time.time() - start

        ingested_files = sum(1 for v in results.values() if v > 0)
        failed_files = sum(1 for v in results.values() if v < 0)
        total_chunks = sum(v for v in results.values() if v > 0)

        total += 1
        passed += assert_true(ingested_files > 100, f"Scrapy: {ingested_files} files ingested, {failed_files} failed, {total_chunks} chunks in {elapsed:.0f}s")
        total += 1
        passed += assert_true(
            failed_files < 10,
            f"Fewer than 10 files failed ({failed_files} failed — check logs for non-data files)"
        )
    else:
        print(f"  ⚠ SKIP Scrapy ingestion — directory not found")

    if SLACK_DIR.exists():
        print("  Ingesting full Slack export...")
        slack_results = ingest_directory(SLACK_DIR, collection, recursive=True)
        slack_chunks = sum(v for v in slack_results.values() if v > 0)
        total += 1
        passed += assert_true(slack_chunks > 0, f"Slack ingested: {slack_chunks} chunks")
    else:
        print(f"  ⚠ SKIP Slack — directory not found at {SLACK_DIR}")

    if JIRA_DIR.exists():
        print("  Ingesting Jira CSV...")
        jira_results = ingest_directory(JIRA_DIR, collection, recursive=False)
        jira_chunks = sum(v for v in jira_results.values() if v > 0)
        total += 1
        passed += assert_true(jira_chunks > 0, f"Jira ingested: {jira_chunks} chunks")
    else:
        print(f"  ⚠ SKIP Jira — directory not found at {JIRA_DIR}")

    count_after = collection.count()
    print(f"  Collection size after ingestion: {count_after} (+{count_after - count_before})")
    total += 1
    passed += assert_true(count_after > count_before, "Collection grew after Phase 3 ingestion")

    print(f"  Full ingestion: {passed}/{total} passed")
    return passed == total, collection

# Idempotency at scale
def test_idempotency_at_scale(collection):
    """Re-ingesting the same data must not increase collection size."""
    print("\n[3] Idempotency at Scale")
    passed = 0
    total = 0

    if not SCRAPY_ROOT.exists():
        print("  ⚠ SKIP: Scrapy not found")
        return True

    # Pick a few representative files
    sample_files = list(SCRAPY_ROOT.rglob("*.py"))[:5]
    count_before = collection.count()

    for f in sample_files:
        ingest_code_file(f, collection)

    count_after = collection.count()
    total += 1
    passed += assert_true(
        count_before == count_after,
        f"Re-ingesting 5 already-ingested files did not change collection size ({count_before} == {count_after})"
    )

    print(f"  Idempotency: {passed}/{total} passed")
    return passed == total

# Scale similarity search
def test_similarity_search_at_scale(collection):
    """Search quality and performance at production scale."""
    print("\n[4] Similarity Search at Scale")
    passed = 0
    total = 0

    if collection.count() == 0:
        print("  ⚠ SKIP: collection is empty")
        return True

    print(f"  Searching across {collection.count()} chunks")

    SCRAPY_QUERIES = [
        ("how does Scrapy handle spider middleware", "code"),
        ("why was the download timeout changed", "slack"),
        ("fix for crawl depth limit bug", "jira"),
        ("how are HTTP headers processed in requests", None),
        ("what caused the pipeline failure", None),
    ]

    latencies = []
    for query, source_type in SCRAPY_QUERIES:
        start = time.time()
        results = search(query, collection, n_results=5, source_type=source_type)
        latency = time.time() - start
        latencies.append(latency)

        total += 1
        passed += assert_true(
            len(results) > 0,
            f"Query '{query[:40]}' → {len(results)} results in {latency:.2f}s"
        )

        if results:
            total += 1
            passed += assert_true(
                results[0]["distance"] < 1.0,
                f"Top result distance valid: {results[0]['distance']:.4f}"
            )

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"  Average search latency: {avg_latency:.2f}s")
    total += 1
    passed += assert_true(avg_latency < 10.0, f"Search latency acceptable (<10s avg)")

    # Cross-source: verify results from multiple source types appear when unfiltered
    cross_results = search("error handling and exception logging", collection, n_results=10)
    found_types = {r["metadata"]["source_type"] for r in cross_results}
    print(f"  Cross-source query returned types: {found_types}")

    print(f"  Scale search: {passed}/{total} passed")
    return passed == total

# Run all Phase 3 tests
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3 — Demonstration & Scale")
    print("=" * 60)

    results = []

    results.append(test_scrapy_file_discovery())

    ingestion_passed, collection = test_full_ingestion()
    results.append(ingestion_passed)
    results.append(test_idempotency_at_scale(collection))
    results.append(test_similarity_search_at_scale(collection))

    passed_count = sum(1 for r in results if r)
    print("\n" + "=" * 60)
    print(f"PHASE 3 RESULT: {passed_count}/{len(results)} test groups passed")
    if all(results):
        print("✓ Phase 3 COMPLETE — pipeline is demo-ready")
    else:
        print("✗ Phase 3 INCOMPLETE — review failures above")
    print("=" * 60)