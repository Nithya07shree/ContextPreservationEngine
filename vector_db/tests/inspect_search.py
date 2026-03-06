import sys
sys.path.insert(0, "C:/ContextEngine")

from ingestor import get_collection, search

collection = get_collection()

query = "whats up with the middleware?" 
results = search(query, collection, n_results=3)

for i, r in enumerate(results, 1):
    print(f"\n{'='*60}")
    print(f"Result {i} — distance: {r['distance']:.4f}")
    print(f"Source: {r['metadata']['source_type']} | {r['metadata']['file_name']}")
    
    # Source-specific context
    if r['metadata']['source_type'] == 'code':
        print(f"Function: {r['metadata']['function_name']} (lines {r['metadata']['start_line']}–{r['metadata']['end_line']})")
    elif r['metadata']['source_type'] == 'slack':
        print(f"Channel: {r['metadata']['channel_name']} | Authors: {r['metadata']['authors']}")
    elif r['metadata']['source_type'] == 'jira':
        print(f"Ticket: {r['metadata']['ticket_id']} | Status: {r['metadata']['status']}")
    
    print(f"\nContent:\n{r['content']}")