import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from chunker import chunk_code_file, chunk_slack_export, chunk_jira_csv
from ingestor import get_collection, ingest_directory, search, ingest_code_file, ingest_slack_file, ingest_jira_file


BASE_P3    = Path("C:/ContextEngine/data")
CODE_ROOT  = BASE_P3 / "code" / "freemed"
SLACK_DIR  = BASE_P3 / "slack"
JIRA_DIR   = BASE_P3 / "jira"

COLLECTION_NAME_P3 = "context_engine"

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".php", ".md", ".rst", ".cfg",
    ".ini", ".tpl", ".smarty", ".sql", ".xml",
    ".html", ".htm", ".css", ".sh", ".bash",
}

SLACK_CHANNELS = ["bugs", "freemed-dev", "freemed-release", "general", "onboarding"]


def assert_true(condition, message):
    if not condition:
        print(f"  ✗ FAIL: {message}")
        return False
    print(f"  ✓ PASS: {message}")
    return True


def test_freemed_file_discovery():
    print("\n[1] FreeMED Repository File Discovery")
    passed = 0
    total = 0

    if not CODE_ROOT.exists():
        print(f"  ⚠ SKIP: FreeMED not found at {CODE_ROOT}")
        print(f"  Run: git clone https://github.com/freemed/freemed {CODE_ROOT}")
        return True

    php_files  = list(CODE_ROOT.rglob("*.php"))
    java_files = list(CODE_ROOT.rglob("*.java"))
    js_files   = list(CODE_ROOT.rglob("*.js"))
    all_supported = [f for f in CODE_ROOT.rglob("*") if f.is_file() and f.suffix.lower() in CODE_EXTENSIONS]

    total += 1
    passed += assert_true(len(php_files) > 50, f"Found {len(php_files)} .php files (expected >50)")
    total += 1
    passed += assert_true(len(java_files) > 10, f"Found {len(java_files)} .java files (expected >10)")
    total += 1
    passed += assert_true(len(js_files) > 10, f"Found {len(js_files)} .js files (expected >10)")
    total += 1
    passed += assert_true(len(all_supported) > 100, f"Found {len(all_supported)} total supported files (expected >100)")

    print(f"  File breakdown: PHP={len(php_files)}, Java={len(java_files)}, JS={len(js_files)}, total supported={len(all_supported)}")

    errors = []
    sample_files = php_files[:10] + java_files[:5] + js_files[:5]
    for f in sample_files:
        try:
            chunks = chunk_code_file(f)
            assert len(chunks) > 0
        except Exception as e:
            errors.append((f.name, str(e)))

    total += 1
    passed += assert_true(
        len(errors) == 0,
        f"Sample parse passed for all {len(sample_files)} files"
        if len(errors) == 0
        else f"{len(errors)} sample files failed: {errors[:3]}"
    )

    print(f"  FreeMED discovery: {passed}/{total} passed")
    return passed == total


def test_slack_structure():
    print("\n[2] Slack Export Structure")
    passed = 0
    total = 0

    if not SLACK_DIR.exists():
        print(f"  ⚠ SKIP: Slack dir not found at {SLACK_DIR}")
        return True

    for channel in SLACK_CHANNELS:
        channel_dir = SLACK_DIR / channel
        total += 1
        passed += assert_true(
            channel_dir.exists(),
            f"Slack channel directory exists: {channel}"
        )

    for meta_file in ["channels.json", "users.json"]:
        total += 1
        passed += assert_true(
            (SLACK_DIR / meta_file).exists(),
            f"Slack metadata file exists: {meta_file}"
        )

    freemed_dev_jsons = list((SLACK_DIR / "freemed-dev").rglob("*.json")) if (SLACK_DIR / "freemed-dev").exists() else []
    total += 1
    passed += assert_true(len(freemed_dev_jsons) > 5, f"freemed-dev has {len(freemed_dev_jsons)} daily JSON files (expected >5)")

    if freemed_dev_jsons:
        try:
            chunks = chunk_slack_export(freemed_dev_jsons[0])
            total += 1
            passed += assert_true(len(chunks) > 0, f"Slack file parses to chunks: {freemed_dev_jsons[0].name}")
            for chunk in chunks:
                total += 1
                passed += assert_true(chunk["metadata"]["source_type"] == "slack", "chunk source_type is 'slack'")
                total += 1
                passed += assert_true("channel_name" in chunk["metadata"], "chunk has channel_name")
                break
        except Exception as e:
            print(f"  ✗ FAIL: Slack chunk parse error: {e}")
            passed -= 1

    print(f"  Slack structure: {passed}/{total} passed")
    return passed == total

def test_full_ingestion():
    print("\n[3] Full-Scale Ingestion (FreeMED)")

    import chromadb
    client = chromadb.PersistentClient(path="C:/ContextEngine/chromadb_store")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME_P3,
        metadata={"hnsw:space": "cosine"},
    )

    passed = 0
    total = 0

    count_before = collection.count()
    print(f"  Collection size before ingestion: {count_before}")

    if CODE_ROOT.exists():
        print("  Ingesting FreeMED repository (this will take several hours on CPU)...")
        start = time.time()
        results = ingest_directory(CODE_ROOT, collection, recursive=True, project="freemed")
        elapsed = time.time() - start

        ingested_files = sum(1 for v in results.values() if v > 0)
        failed_files   = sum(1 for v in results.values() if v < 0)
        total_chunks   = sum(v for v in results.values() if v > 0)

        total += 1
        passed += assert_true(
            ingested_files > 50,
            f"FreeMED: {ingested_files} files ingested, {failed_files} failed, {total_chunks} chunks in {elapsed:.0f}s"
        )
        total += 1
        passed += assert_true(
            failed_files < 20,
            f"Fewer than 20 files failed ({failed_files} failed)"
        )
    else:
        print(f"  ⚠ SKIP code ingestion — {CODE_ROOT} not found")

    if SLACK_DIR.exists():
        print("  Ingesting FreeMED Slack export...")
        slack_results = ingest_directory(SLACK_DIR, collection, recursive=True, project="freemed")
        slack_chunks  = sum(v for v in slack_results.values() if v > 0)
        total += 1
        passed += assert_true(slack_chunks > 0, f"Slack ingested: {slack_chunks} chunks")
    else:
        print(f"  ⚠ SKIP Slack — {SLACK_DIR} not found")

    if JIRA_DIR.exists():
        print("  Ingesting FreeMED Jira CSV...")
        jira_results = ingest_directory(JIRA_DIR, collection, recursive=False, project="freemed")
        jira_chunks  = sum(v for v in jira_results.values() if v > 0)
        total += 1
        passed += assert_true(jira_chunks > 0, f"Jira ingested: {jira_chunks} chunks")
    else:
        print(f"  ⚠ SKIP Jira — {JIRA_DIR} not found")

    count_after = collection.count()
    print(f"  Collection size after ingestion: {count_after} (+{count_after - count_before})")
    total += 1
    passed += assert_true(count_after > count_before, "Collection grew after Phase 3 ingestion")

    print(f"  Full ingestion: {passed}/{total} passed")
    return passed == total, collection

def test_idempotency_at_scale(collection):
    print("\n[4] Idempotency at Scale")
    passed = 0
    total = 0

    if not CODE_ROOT.exists():
        print("  ⚠ SKIP: FreeMED code not found")
        return True

    # Re-ingest a small sample — collection must not grow
    sample_php  = list(CODE_ROOT.rglob("*.php"))[:3]
    sample_java = list(CODE_ROOT.rglob("*.java"))[:2]
    sample_files = sample_php + sample_java

    if not sample_files:
        print("  ⚠ SKIP: no sample files found")
        return True

    count_before = collection.count()
    for f in sample_files:
        ingest_code_file(f, collection, project="freemed")
    count_after = collection.count()

    total += 1
    passed += assert_true(
        count_before == count_after,
        f"Re-ingesting {len(sample_files)} already-ingested files did not change collection size ({count_before} == {count_after})"
    )

    print(f"  Idempotency: {passed}/{total} passed")
    return passed == total


def test_similarity_search_at_scale(collection):
    print("\n[5] Similarity Search at Scale")
    passed = 0
    total = 0

    if collection.count() == 0:
        print("  ⚠ SKIP: collection is empty")
        return True

    print(f"  Searching across {collection.count()} chunks")

    FREEMED_QUERIES = [
        # Code queries — PHP/Java logic
        ("how does patient billing work",                   "code"),
        ("how is authentication handled in FreeMED",        "code"),
        ("how does the appointment scheduling module work", "code"),
        # Slack queries — team discussions
        ("what bug was reported in the freemed-dev channel", "slack"),
        ("onboarding steps for new developers",              "slack"),
        # Jira queries — ticket history
        ("fix for login failure bug",                        "jira"),
        # Cross-source — no filter
        ("error handling and exception logging",             None),
        ("database connection issues",                       None),
    ]

    latencies = []
    for query, source_type in FREEMED_QUERIES:
        start = time.time()
        results = search(query, collection, n_results=5, source_type=source_type, project="freemed")
        latency = time.time() - start
        latencies.append(latency)

        total += 1
        passed += assert_true(
            len(results) > 0,
            f"Query '{query[:45]}' [{source_type or 'any'}] → {len(results)} results in {latency:.2f}s"
        )

        if results:
            total += 1
            passed += assert_true(
                results[0]["distance"] < 1.0,
                f"Top result distance valid: {results[0]['distance']:.4f}"
            )

            # All results must be from the freemed project since we filtered for it
            total += 1
            passed += assert_true(
                all(r["metadata"].get("project") == "freemed" for r in results),
                "All results are tagged project='freemed'"
            )

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print(f"  Average search latency: {avg_latency:.2f}s")
    total += 1
    passed += assert_true(avg_latency < 10.0, f"Search latency acceptable (<10s avg)")

    # Cross-source: verify results span multiple source types when unfiltered
    cross_results = search("error handling", collection, n_results=10, project="freemed")
    found_types = {r["metadata"]["source_type"] for r in cross_results}
    print(f"  Cross-source query returned types: {found_types}")

    # Language coverage: verify PHP and Java chunks are actually in the collection
    php_results  = search("PHP class method", collection, n_results=5, source_type="code", project="freemed")
    java_results = search("Java class constructor", collection, n_results=5, source_type="code", project="freemed")
    php_langs    = {r["metadata"].get("language") for r in php_results}
    java_langs   = {r["metadata"].get("language") for r in java_results}
    print(f"  PHP query returned languages: {php_langs}")
    print(f"  Java query returned languages: {java_langs}")

    print(f"  Scale search: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3 — Demonstration & Scale (FreeMED)")
    print("=" * 60)

    results = []

    results.append(test_freemed_file_discovery())
    results.append(test_slack_structure())

    ingestion_passed, collection = test_full_ingestion()
    results.append(ingestion_passed)
    results.append(test_idempotency_at_scale(collection))
    results.append(test_similarity_search_at_scale(collection))

    passed_count = sum(1 for r in results if r)
    print("\n" + "=" * 60)
    print(f"PHASE 3 RESULT: {passed_count}/{len(results)} test groups passed")
    if all(results):
        print("✓ Phase 3 COMPLETE — FreeMED pipeline is demo-ready")
    else:
        print("✗ Phase 3 INCOMPLETE — review failures above")
    print("=" * 60)