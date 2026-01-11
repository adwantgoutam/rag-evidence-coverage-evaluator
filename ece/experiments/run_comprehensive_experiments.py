"""Comprehensive experiment script for ECE evaluation.

Author: Goutam Adwant (gadwant)"""

import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ece.evaluator import EvidenceCoverageEvaluator
from ece.models import Context, Passage
from ece.ollama_judge import OllamaJudge
from ece.claim_extractor import ClaimExtractor
from ece.evidence_retriever import EvidenceRetriever


def create_test_dataset():
    """Create a comprehensive test dataset."""
    test_cases = [
        {
            "name": "Eiffel Tower",
            "answer": "The Eiffel Tower is located in Paris, France. It was completed in 1889 and stands 330 meters tall. The tower was designed by Gustave Eiffel and is one of the most visited monuments in the world, attracting millions of tourists annually.",
            "context": {
                "passages": [
                    {"id": "p1", "text": "The Eiffel Tower is an iron lattice tower located on the Champ de Mars in Paris, France. It was named after the engineer Gustave Eiffel, whose company designed and built the tower."},
                    {"id": "p2", "text": "The Eiffel Tower was constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair. It was initially criticized by some of France's leading artists and intellectuals for its design, but has become a global cultural icon of France."},
                    {"id": "p3", "text": "The tower is 330 meters (1,083 feet) tall, about the same height as an 81-story building. It was the tallest man-made structure in the world from 1889 until the completion of the Chrysler Building in New York in 1930."},
                    {"id": "p4", "text": "The Eiffel Tower is the most-visited paid monument in the world, with over 6 million visitors annually. It has been a UNESCO World Heritage Site since 1991."}
                ]
            }
        },
        {
            "name": "Quantum Computing",
            "answer": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement. Quantum computers can solve certain problems much faster than classical computers. However, quantum computers are still in early development stages.",
            "context": {
                "passages": [
                    {"id": "p1", "text": "Quantum computing is a type of computation that uses quantum mechanical phenomena such as superposition and quantum entanglement to perform calculations."},
                    {"id": "p2", "text": "Quantum computers can theoretically solve certain problems exponentially faster than classical computers, particularly in cryptography and optimization."},
                    {"id": "p3", "text": "Current quantum computers are still in the early stages of development, with most systems having fewer than 100 qubits and requiring extremely low temperatures to operate."}
                ]
            }
        }
    ]
    return test_cases


def run_mode_a_experiment(answer: str, context: Context, retrieval_method: str):
    """Run Mode A experiment."""
    evaluator = EvidenceCoverageEvaluator(
        retrieval_method=retrieval_method,
        retrieval_top_k=3,
        nli_model="roberta-large-mnli",
        threshold=0.7
    )
    start_time = time.time()
    result = evaluator.evaluate(answer, context)
    elapsed = time.time() - start_time
    
    return {
        "coverage_score": result.coverage_score,
        "total_claims": result.total_claims,
        "supported_claims": result.supported_claims,
        "time": elapsed
    }


def run_mode_b_experiment(answer: str, context: Context, model_name: str):
    """Run Mode B experiment with Ollama."""
    try:
        claim_extractor = ClaimExtractor()
        claims = claim_extractor.extract_claims(answer)
        
        evidence_retriever = EvidenceRetriever(method="bm25", top_k=3)
        evidence_retriever.index_passages(context.passages)
        
        judge = OllamaJudge(model=model_name, temperature=0.0)
        
        start_time = time.time()
        claim_analyses = []
        for claim in claims:
            evidence = evidence_retriever.retrieve(claim)
            analysis = judge.score_claim(claim, evidence, threshold=0.7)
            claim_analyses.append(analysis)
        
        supported_count = sum(1 for a in claim_analyses if a.supported)
        coverage_score = supported_count / len(claims) if claims else 0.0
        elapsed = time.time() - start_time
        
        return {
            "coverage_score": coverage_score,
            "total_claims": len(claims),
            "supported_claims": supported_count,
            "time": elapsed
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    """Run comprehensive experiments."""
    test_cases = create_test_dataset()
    all_results = {}
    
    print("Running comprehensive ECE experiments...")
    
    for test_case in test_cases:
        name = test_case["name"]
        answer = test_case["answer"]
        passages = [Passage(id=p["id"], text=p["text"]) for p in test_case["context"]["passages"]]
        context = Context(passages=passages)
        
        print(f"\n=== Test Case: {name} ===")
        
        # Mode A experiments
        print("Mode A - BM25...")
        all_results[f"{name}_mode_a_bm25"] = run_mode_a_experiment(answer, context, "bm25")
        
        print("Mode A - Embedding...")
        all_results[f"{name}_mode_a_embedding"] = run_mode_a_experiment(answer, context, "embedding")
        
        # Mode B experiments
        models = ["mistral:latest", "llama3:latest", "gemma3:latest", "deepseek-r1:latest"]
        for model in models:
            print(f"Mode B - {model}...")
            result = run_mode_b_experiment(answer, context, model)
            all_results[f"{name}_mode_b_{model.replace(':', '_')}"] = result
    
    # Save results
    results_file = Path("experiment_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    return all_results


if __name__ == "__main__":
    main()

