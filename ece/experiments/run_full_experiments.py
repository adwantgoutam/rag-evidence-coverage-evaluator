"""Full experiment script using actual ECE code with Ollama.

Author: Goutam Adwant (gadwant)"""

import json
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ece.evaluator import EvidenceCoverageEvaluator
    from ece.models import Context, Passage
    from ece.ollama_judge import OllamaJudge
    from ece.claim_extractor import ClaimExtractor
    from ece.evidence_retriever import EvidenceRetriever
    from ece.nli_scorer import NLIScorer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


def load_test_data():
    """Load test data."""
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


def run_mode_a_bm25(answer, context):
    """Run Mode A with BM25."""
    print("Running Mode A (BM25)...")
    evaluator = EvidenceCoverageEvaluator(
        retrieval_method="bm25",
        retrieval_top_k=3,
        nli_model="roberta-large-mnli",
        threshold=0.7
    )
    start = time.time()
    result = evaluator.evaluate(answer, context)
    elapsed = time.time() - start
    
    return {
        "coverage_score": result.coverage_score,
        "total_claims": result.total_claims,
        "supported_claims": result.supported_claims,
        "time": elapsed
    }


def run_mode_a_embedding(answer, context):
    """Run Mode A with Embedding."""
    print("Running Mode A (Embedding)...")
    evaluator = EvidenceCoverageEvaluator(
        retrieval_method="embedding",
        retrieval_top_k=3,
        nli_model="roberta-large-mnli",
        threshold=0.7
    )
    start = time.time()
    result = evaluator.evaluate(answer, context)
    elapsed = time.time() - start
    
    return {
        "coverage_score": result.coverage_score,
        "total_claims": result.total_claims,
        "supported_claims": result.supported_claims,
        "time": elapsed
    }


def run_mode_b_ollama(answer, context, model_name):
    """Run Mode B with Ollama model."""
    print(f"Running Mode B ({model_name})...")
    try:
        claim_extractor = ClaimExtractor()
        claims = claim_extractor.extract_claims(answer)
        
        evidence_retriever = EvidenceRetriever(method="bm25", top_k=3)
        evidence_retriever.index_passages(context.passages)
        
        judge = OllamaJudge(model=model_name, temperature=0.0)
        
        start = time.time()
        claim_analyses = []
        success_count = 0
        
        for claim in claims:
            try:
                evidence = evidence_retriever.retrieve(claim)
                analysis = judge.score_claim(claim, evidence, threshold=0.7)
                claim_analyses.append(analysis)
                success_count += 1
            except Exception as e:
                print(f"  Warning: Error processing claim: {e}")
                # Create unsupported analysis
                from ece.models import ClaimAnalysis
                claim_analyses.append(ClaimAnalysis(
                    claim=claim,
                    supported=False,
                    support_score=0.0,
                    supporting_snippets=[],
                    missing_info=f"Error: {str(e)}"
                ))
        
        supported_count = sum(1 for a in claim_analyses if a.supported)
        coverage_score = supported_count / len(claims) if claims else 0.0
        elapsed = time.time() - start
        
        return {
            "coverage_score": coverage_score,
            "total_claims": len(claims),
            "supported_claims": supported_count,
            "time": elapsed,
            "success_rate": success_count / len(claims) if claims else 0.0
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    """Run all experiments."""
    print("=" * 60)
    print("ECE Comprehensive Experiments")
    print("=" * 60)
    
    # Load data
    print("\nLoading test data...")
    answer, context = load_test_data()
    print(f"Answer length: {len(answer)} chars")
    print(f"Context passages: {len(context.passages)}")
    
    all_results = {}
    
    # Mode A experiments
    print("\n" + "=" * 60)
    print("MODE A EXPERIMENTS")
    print("=" * 60)
    
    try:
        result_bm25 = run_mode_a_bm25(answer, context)
        all_results["mode_a_bm25"] = result_bm25
        print(f"  Coverage: {result_bm25['coverage_score']:.3f}, Time: {result_bm25['time']:.2f}s")
    except Exception as e:
        print(f"  Error: {e}")
        all_results["mode_a_bm25"] = {"error": str(e)}
    
    try:
        result_embed = run_mode_a_embedding(answer, context)
        all_results["mode_a_embedding"] = result_embed
        print(f"  Coverage: {result_embed['coverage_score']:.3f}, Time: {result_embed['time']:.2f}s")
    except Exception as e:
        print(f"  Error: {e}")
        all_results["mode_a_embedding"] = {"error": str(e)}
    
    # Mode B experiments
    print("\n" + "=" * 60)
    print("MODE B EXPERIMENTS (Ollama)")
    print("=" * 60)
    
    models = ["mistral:latest", "llama3:latest", "gemma3:latest", "deepseek-r1:latest"]
    for model in models:
        try:
            result = run_mode_b_ollama(answer, context, model)
            model_key = model.replace(":", "_")
            all_results[f"mode_b_{model_key}"] = result
            if "error" not in result:
                print(f"  {model}: Coverage={result['coverage_score']:.3f}, Time={result['time']:.2f}s, Success={result.get('success_rate', 1.0):.2%}")
            else:
                print(f"  {model}: Error - {result['error']}")
        except Exception as e:
            print(f"  {model}: Exception - {e}")
            all_results[f"mode_b_{model.replace(':', '_')}"] = {"error": str(e)}
    
    # Save results
    results_file = Path("experiment_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Results saved to {results_file}")
    print("=" * 60)
    
    # Print summary
    print("\nSummary:")
    for key, value in all_results.items():
        if "error" not in value:
            print(f"  {key}: Coverage={value.get('coverage_score', 0):.3f}, Time={value.get('time', 0):.2f}s")
    
    return all_results


if __name__ == "__main__":
    main()

