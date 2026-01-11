"""Script to run ECE experiments with Ollama models.

Author: Goutam Adwant (gadwant)"""

import json
import time
from pathlib import Path
from typing import Dict, List
from ece.evaluator import EvidenceCoverageEvaluator
from ece.models import Context, Passage
from ece.ollama_judge import OllamaJudge
from ece.claim_extractor import ClaimExtractor
from ece.evidence_retriever import EvidenceRetriever
from ece.nli_scorer import NLIScorer


def load_test_data():
    """Load test data from examples."""
    examples_dir = Path("examples")
    
    # Load answer
    answer_file = examples_dir / "example_answer.txt"
    with open(answer_file, "r") as f:
        answer = f.read()
    
    # Load context
    context_file = examples_dir / "example_context.json"
    with open(context_file, "r") as f:
        context_data = json.load(f)
    
    passages = [Passage(id=p["id"], text=p["text"]) for p in context_data["passages"]]
    context = Context(passages=passages)
    
    return answer, context


def run_mode_a_experiments(answer: str, context: Context) -> Dict:
    """Run Mode A (NLI-based) experiments."""
    results = {}
    
    print("\n=== Mode A Experiments ===")
    
    # Test with BM25 retrieval
    print("\n1. Mode A with BM25 retrieval...")
    start_time = time.time()
    evaluator = EvidenceCoverageEvaluator(
        retrieval_method="bm25",
        retrieval_top_k=3,
        nli_model="roberta-large-mnli",
        threshold=0.7
    )
    result_bm25 = evaluator.evaluate(answer, context)
    time_bm25 = time.time() - start_time
    
    results["mode_a_bm25"] = {
        "coverage_score": result_bm25.coverage_score,
        "total_claims": result_bm25.total_claims,
        "supported_claims": result_bm25.supported_claims,
        "time": time_bm25
    }
    
    # Test with Embedding retrieval (using nomic-embed-text via sentence-transformers)
    print("\n2. Mode A with Embedding retrieval...")
    start_time = time.time()
    evaluator = EvidenceCoverageEvaluator(
        retrieval_method="embedding",
        retrieval_top_k=3,
        nli_model="roberta-large-mnli",
        threshold=0.7
    )
    result_embed = evaluator.evaluate(answer, context)
    time_embed = time.time() - start_time
    
    results["mode_a_embedding"] = {
        "coverage_score": result_embed.coverage_score,
        "total_claims": result_embed.total_claims,
        "supported_claims": result_embed.supported_claims,
        "time": time_embed
    }
    
    return results


def run_mode_b_experiments(answer: str, context: Context) -> Dict:
    """Run Mode B (LLM-based) experiments with Ollama models."""
    results = {}
    
    print("\n=== Mode B Experiments ===")
    
    models = ["mistral:latest", "llama3:latest", "gemma3:latest", "deepseek-r1:latest"]
    
    # Extract claims and retrieve evidence first (shared across models)
    claim_extractor = ClaimExtractor()
    claims = claim_extractor.extract_claims(answer)
    
    evidence_retriever = EvidenceRetriever(method="bm25", top_k=3)
    evidence_retriever.index_passages(context.passages)
    
    evidence_dict = {}
    for claim in claims:
        evidence = evidence_retriever.retrieve(claim)
        evidence_dict[claim] = evidence
    
    # Test each Ollama model
    for model_name in models:
        print(f"\nTesting {model_name}...")
        try:
            judge = OllamaJudge(model=model_name, temperature=0.0)
            
            start_time = time.time()
            claim_analyses = []
            
            for claim in claims:
                evidence = evidence_dict[claim]
                analysis = judge.score_claim(claim, evidence, threshold=0.7)
                claim_analyses.append(analysis)
            
            # Calculate coverage
            supported_count = sum(1 for a in claim_analyses if a.supported)
            coverage_score = supported_count / len(claims) if claims else 0.0
            time_taken = time.time() - start_time
            
            results[f"mode_b_{model_name.replace(':', '_')}"] = {
                "coverage_score": coverage_score,
                "total_claims": len(claims),
                "supported_claims": supported_count,
                "time": time_taken
            }
            
            print(f"  Coverage: {coverage_score:.2f}, Time: {time_taken:.2f}s")
            
        except Exception as e:
            print(f"  Error with {model_name}: {e}")
            results[f"mode_b_{model_name.replace(':', '_')}"] = {
                "error": str(e)
            }
    
    return results


def main():
    """Run all experiments."""
    print("Loading test data...")
    answer, context = load_test_data()
    
    all_results = {}
    
    # Run Mode A experiments
    mode_a_results = run_mode_a_experiments(answer, context)
    all_results.update(mode_a_results)
    
    # Run Mode B experiments
    mode_b_results = run_mode_b_experiments(answer, context)
    all_results.update(mode_b_results)
    
    # Save results
    results_file = Path("experiment_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n=== Results saved to {results_file} ===")
    print("\nSummary:")
    for key, value in all_results.items():
        if "error" not in value:
            print(f"{key}: Coverage={value.get('coverage_score', 0):.2f}, Time={value.get('time', 0):.2f}s")


if __name__ == "__main__":
    main()

