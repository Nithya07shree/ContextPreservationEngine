# Test phase 2 runs on 3 python files from scrapy, and generated slack and jira files.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from chunker import chunk_code_file, chunk_slack_export, chunk_jira_csv, MAX_CODE_LINES, MAX_SLACK_TOKENS, AVG_CHARS_PER_TOKEN
from ingestor import get_collection, ingest_code_file, ingest_slack_file, ingest_jira_file, search, ingest_directory

BASE_P1 = Path("C:/ContextEngine/test_data/phase1")
BASE_P2 = Path("C:/ContextEngine/test_data/phase2")

COLLECTION_NAME_TEST = "test_phase2"


def assert_true(condition, message):
    if not condition:
        print(f"  ✗ FAIL: {message}")
        return False
    print(f"  ✓ PASS: {message}")
    return True


def test_phase1_regression():
    print("\n[REGRESSION] Phase 1 data still ingests cleanly")
    import chromadb
    client = chromadb.PersistentClient(path="C:/ContextEngine/chromadb_store")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME_TEST,
        metadata={"hnsw:space": "cosine"},
    )

    passed = 0
    total = 0

    for code_file in (BASE_P1 / "code").glob("*.py"):
        count = ingest_code_file(code_file, collection)
        total += 1
        passed += assert_true(count > 0, f"Phase 1 code re-ingests: {code_file.name}")

    for slack_file in (BASE_P1 / "slack").glob("*.json"):
        count = ingest_slack_file(slack_file, collection)
        total += 1
        passed += assert_true(count > 0, f"Phase 1 slack re-ingests: {slack_file.name}")

    for jira_file in (BASE_P1 / "jira").glob("*.csv"):
        count = ingest_jira_file(jira_file, collection)
        total += 1
        passed += assert_true(count > 0, f"Phase 1 jira re-ingests: {jira_file.name}")

    print(f"  Regression: {passed}/{total} passed")
    return passed == total, collection

# Chunk size validation
def test_chunk_size_limits():
    print("\n[1] Chunk Size Limit Validation")
    passed = 0
    total = 0

    code_files = list((BASE_P2 / "code").glob("*.py"))
    if not code_files:
        print("  ⚠ SKIP: no Phase 2 code files found")
    else:
        all_chunks = []
        for f in code_files:
            all_chunks.extend(chunk_code_file(f))

        oversized = [
            c for c in all_chunks
            if (c["content"].count("\n") + 1) > MAX_CODE_LINES + 5
        ]
        total += 1
        passed += assert_true(
            len(oversized) == 0,
            f"No code chunks exceed {MAX_CODE_LINES} lines (checked {len(all_chunks)} chunks)"
        )

        # Verify split chunks have correct chunk_index / total_chunks
        split_chunks = [c for c in all_chunks if c["metadata"]["total_chunks"] > 1]
        for chunk in split_chunks:
            total += 1
            passed += assert_true(
                chunk["metadata"]["chunk_index"] < chunk["metadata"]["total_chunks"],
                f"Split chunk index valid: {chunk['metadata']['chunk_index']} < {chunk['metadata']['total_chunks']}"
            )
            break 

    slack_files = list((BASE_P2 / "slack").glob("*.json"))
    if not slack_files:
        print("  ⚠ SKIP: no Phase 2 slack files found")
    else:
        for f in slack_files:
            chunks = chunk_slack_export(f)
            max_chars = MAX_SLACK_TOKENS * AVG_CHARS_PER_TOKEN
            oversized_slack = [
                c for c in chunks
                if len(c["content"]) > max_chars * 1.1  # 10% tolerance
            ]
            total += 1
            passed += assert_true(
                len(oversized_slack) == 0,
                f"No slack chunks exceed token limit ({len(chunks)} total chunks in {f.name})"
            )

    print(f"  Chunk size limits: {passed}/{total} passed")
    return passed == total

# Mixed file type parsing
def test_mixed_file_parsing():
    print("\n[2] Mixed File Parsing")
    passed = 0
    total = 0

    code_files = list((BASE_P2 / "code").glob("*.py"))
    total += 1
    passed += assert_true(len(code_files) >= 3, f"Found {len(code_files)} code files (need ≥3)")

    for f in code_files:
        try:
            chunks = chunk_code_file(f)
            total += 1
            passed += assert_true(len(chunks) > 0, f"{f.name}: produced {len(chunks)} chunks")
        except Exception as e:
            total += 1
            passed += assert_true(False, f"{f.name}: raised exception — {e}")

    slack_files = list((BASE_P2 / "slack").glob("*.json"))
    for f in slack_files:
        try:
            chunks = chunk_slack_export(f)
            total += 1
            passed += assert_true(len(chunks) > 0, f"{f.name}: produced {len(chunks)} slack chunks")
            threaded = [c for c in chunks if c["content"].count("[") > 1]
            total += 1
            passed += assert_true(
                len(threaded) > 0,
                f"{f.name}: at least one thread chunk contains multiple messages"
            )
        except Exception as e:
            total += 1
            passed += assert_true(False, f"{f.name}: raised exception — {e}")

    jira_files = list((BASE_P2 / "jira").glob("*.csv"))
    for f in jira_files:
        try:
            chunks = chunk_jira_csv(f)
            total += 1
            passed += assert_true(len(chunks) >= 25, f"{f.name}: ≥25 Jira chunks (got {len(chunks)})")
        except Exception as e:
            total += 1
            passed += assert_true(False, f"{f.name}: raised exception — {e}")

    print(f"  Mixed file parsing: {passed}/{total} passed")
    return passed == total

# Metadata completeness
def test_metadata_completeness():
    print("\n[3] Metadata Completeness")
    passed = 0
    total = 0

    REQUIRED_ALL = {"source_type", "file_name", "chunk_index", "total_chunks"}
    REQUIRED_CODE = {"function_name", "language", "start_line", "end_line"}
    REQUIRED_SLACK = {"channel_name", "timestamp", "authors"}
    REQUIRED_JIRA = {"ticket_id"}

    def check_fields(chunk, required, label):
        missing = required - set(chunk["metadata"].keys())
        return assert_true(len(missing) == 0, f"{label}: has required fields (missing: {missing})")

    code_files = list((BASE_P2 / "code").glob("*.py"))
    for f in code_files[:1]:
        for chunk in chunk_code_file(f)[:3]:
            total += 1
            passed += check_fields(chunk, REQUIRED_ALL | REQUIRED_CODE, f"code/{f.name}")

    slack_files = list((BASE_P2 / "slack").glob("*.json"))
    for f in slack_files[:1]:
        for chunk in chunk_slack_export(f)[:3]:
            total += 1
            passed += check_fields(chunk, REQUIRED_ALL | REQUIRED_SLACK, f"slack/{f.name}")

    jira_files = list((BASE_P2 / "jira").glob("*.csv"))
    for f in jira_files[:1]:
        for chunk in chunk_jira_csv(f)[:3]:
            total += 1
            passed += check_fields(chunk, REQUIRED_ALL | REQUIRED_JIRA, f"jira/{f.name}")

    print(f"  Metadata completeness: {passed}/{total} passed")
    return passed == total

# Phase 2 similarity search
def test_similarity_search_phase2(collection):
    print("\n[4] Phase 2 Similarity Search")

    for f in (BASE_P2 / "code").glob("*.py"):
        ingest_code_file(f, collection)
    for f in (BASE_P2 / "slack").glob("*.json"):
        ingest_slack_file(f, collection)
    for f in (BASE_P2 / "jira").glob("*.csv"):
        ingest_jira_file(f, collection)

    passed = 0
    total = 0

    queries = [
        ("explain the spider crawling logic", "code"),
        ("why was this feature added", "slack"),
        ("what bug was fixed in this ticket", "jira"),
        ("request handling middleware", None), 
    ]

    for query, source_type in queries:
        results = search(query, collection, n_results=3, source_type=source_type)
        total += 1
        passed += assert_true(
            len(results) > 0,
            f"Query '{query[:40]}' (filter={source_type}) → {len(results)} results"
        )
        if results:
            total += 1
            passed += assert_true(
                results[0]["distance"] < 1.0,
                f"Top result has valid cosine distance: {results[0]['distance']:.4f}"
            )

    print(f"  Phase 2 search: {passed}/{total} passed")
    return passed == total

# Run all Phase 2 tests
if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2 — Chunking Robustness")
    print("=" * 60)

    results = []

    regression_passed, collection = test_phase1_regression()
    results.append(regression_passed)
    results.append(test_chunk_size_limits())
    results.append(test_mixed_file_parsing())
    results.append(test_metadata_completeness())
    results.append(test_similarity_search_phase2(collection))

    passed_count = sum(1 for r in results if r)
    print("\n" + "=" * 60)
    print(f"PHASE 2 RESULT: {passed_count}/{len(results)} test groups passed")
    if all(results):
        print("✓ Phase 2 COMPLETE — ready for Phase 3")
    else:
        print("✗ Phase 2 INCOMPLETE — fix failures before proceeding")
    print("=" * 60)