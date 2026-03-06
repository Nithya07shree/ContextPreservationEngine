import ollama
import time

def test_ollama_conn():
    try:
        models = ollama.list()
        for model in models['models']:
            print(f"Ollama is running. Available models: {len(models['models'])}")
        return True
    except Exception as e:
        print(f"Ollama connection failed {e}")
        return False

def ollama_inference():
    model_name = "qwen2.5-coder:3b"
    try:
        print(f"Testing model inference for {model_name}")
        start = time.time()
        response = ollama.chat(
            model = model_name,
            messages=[{
                'role':'user',
                'content':'Explain vector database in one sentence'
            }]
        )
        time_taken = time.time() - start
        
        answer = response['message']['content']
        print(f"Response time: {time_taken:.2f}s")
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Ollama inference for {model_name} failed with exception {e}")
        
if __name__ == "__main__":
    if not test_ollama_conn():
        exit(1)
    print("ollama connection passed")
    if not ollama_inference():
        exit(1)
    print("inference test passed")