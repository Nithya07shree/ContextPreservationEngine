import sys
sys.path.insert(0, "C:/ContextEngine")

import chromadb
from config import CHROMA_PATH

client = chromadb.PersistentClient(path=CHROMA_PATH)

print("Collections found:")
for col in client.list_collections():
    print(f"  {col.name} ({col.count()} chunks)")

TEST_COLLECTIONS = ["test_phase1", "test_phase2"] 

for name in TEST_COLLECTIONS:
    try:
        client.delete_collection(name)
        print(f"Deleted: {name}")
    except Exception:
        print(f"Not found (skipping): {name}")