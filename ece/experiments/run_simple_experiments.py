"""Simple experiment script that works without full dependencies.

Author: Goutam Adwant (gadwant)"""

import json
import time
import requests
from pathlib import Path

def test_ollama_model(model_name):
    """Test an Ollama model with a simple prompt."""
    try:
        prompt = """You are evaluating whether a claim is supported by evidence.

CLAIM: The Eiffel Tower is located in Paris.

EVIDENCE:
[Passage 1]: The Eiffel Tower is an iron lattice tower located on the Champ de Mars in Paris, France.

Evaluate whether the claim is FULLY supported by the provided evidence. Respond in JSON format:
{"supported": true/false, "confidence": 0.0-1.0, "supporting_texts": ["quotes"], "missing_info": "description"}

Respond ONLY with valid JSON, no other text."""

        start_time = time.time()
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            },
            timeout=60
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "time": elapsed,
                "response_length": len(result.get("response", ""))
            }
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_experiments():
    """Run simple experiments to test Ollama models."""
    models = ["mistral:latest", "llama3:latest", "gemma3:latest", "deepseek-r1:latest"]
    results = {}
    
    print("Testing Ollama models...")
    for model in models:
        print(f"\nTesting {model}...")
        result = test_ollama_model(model)
        results[model] = result
        if result.get("success"):
            print(f"  ✓ Success - Time: {result['time']:.2f}s")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
    
    # Save results
    results_file = Path("ollama_test_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    return results


if __name__ == "__main__":
    run_experiments()

