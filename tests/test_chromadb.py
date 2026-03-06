"""
check for chromadb connection from chroma server, hosted on chromaclient server
"""
import chromadb

try:
    client = chromadb.HttpClient(
        host="192.168.159.101",
        port=8000
    )
    
    heartbeat = client.heartbeat()
    print("chromadb server is running")
    print(f"Heartbeat: {heartbeat}")
    
    collections = client.list_collections()
    print("Collections in database: ")
    
    if collections:
        for c in collections:
            print("-", c.name)
    else:
        print("No collection found")

except Exception as e:
    print(f"failed to connect to chromaDB, {e}")
    
    
