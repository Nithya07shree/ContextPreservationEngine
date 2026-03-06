import requests

response = requests.post("http://localhost:11434/api/embeddings", json={
    "model": "qllama/bge-m3:q8_0",
    "prompt": "Why does the payment retry logic exist?"
})

data = response.json()
embedding = data["embedding"]
print(f"Model: qllama/bge-m3:q8_0")
print(f"Dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

'''
Output:

Model: qllama/bge-m3:q8_0
Dimensions: 1024
First 5 values: [-0.764039933681488, 0.13746017217636108, 0.15188537538051605, -0.03135692700743675, 0.2952035665512085]
'''

response = requests.post("http://localhost:11434/api/embeddings", json={
    "model": "qllama/bge-m3:q4_k_m",
    "prompt": "Why does the payment retry logic exist?"
})

data = response.json()
embedding = data["embedding"]
print(f"Model: qllama/bge-m3:q4_k_m")
print(f"Dimensions: {len(embedding)}")

'''
Output:

Model: qllama/bge-m3:q4_k_m
Dimensions: 1024
'''

import psutil

def select_embedding_model() -> str:
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    print(f"Available RAM: {available_gb:.2f} GB")
    if available_gb >= 4.0:  # q8_0 needs ~3.5GB headroom
        model = "qllama/bge-m3:q8_0"
    else:
        model = "qllama/bge-m3:q4_k_m"
    print(f"Selected model: {model}")
    return model

select_embedding_model()

'''

Output on a machine with 16GB RAM and ollama running (With other background applications, like chrome):
Available RAM: 3.02 GB
Selected model: qllama/bge-m3:q4_k_m

Output on a machine with 16GB RAM with no ollama (With other background applications, like chrome):
Available RAM: 5.07 GB
Selected model: qllama/bge-m3:q8_0

'''