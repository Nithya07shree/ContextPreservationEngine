import sys
sys.path.insert(0, "C:/ContextEngine")

import chromadb 
from pprint import pprint
from config import CHROMA_PATH, COLLECTION_NAME

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

print(f"Collection name: {collection.name}")
print(f"Collection metadata: {collection.metadata}")
pprint(collection.peek(5))

print(f"Total chunks in collection: {collection.count()}")

# results = collection.get(include=["metadatas"])
# from collections import Counter
# source_counts = Counter(m["source_type"] for m in results["metadatas"])
# file_counts = Counter(m["file_name"] for m in results["metadatas"])

# print("\nBy source type:")
# for source, count in sorted(source_counts.items()):
#     print(f"  {source}: {count} chunks")

# print("\nBy file:")
# for file, count in sorted(file_counts.items()):
#     print(f"  {file}: {count} chunks")

# print("\nSample chunks (first 3):")
# sample = collection.get(limit=3, include=["documents", "metadatas"])
# for doc, meta in zip(sample["documents"], sample["metadatas"]):
#     print(f"\n  [{meta['source_type']}] {meta['file_name']}")
#     print(f"  {doc[:200]}...")