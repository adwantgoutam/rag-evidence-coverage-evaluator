# ECE Experiments

This folder contains experiment scripts for evaluating ECE with Ollama models.

## Prerequisites

1. **Ollama installed and running**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   ```

2. **Required models downloaded**
   - mistral:latest
   - llama3:latest
   - gemma3:latest
   - deepseek-r1:latest
   - nomic-embed-text:latest (for embedding retrieval)

3. **Python dependencies**
   ```bash
   pip install requests
   ```

## Running Experiments

### Basic Experiment
```bash
python run_comprehensive_experiments.py
```

This will:
- Run Mode A experiments (BM25 and Embedding retrieval)
- Run Mode B experiments with all Ollama models
- Save results to `experiment_results.json`

### Expected Output
- Coverage scores for each configuration
- Processing times
- Success rates for Mode B
- Comparison metrics

## Results Format

Results are saved as JSON with the following structure:
```json
{
  "test_case_mode_a_bm25": {
    "coverage_score": 0.88,
    "total_claims": 12,
    "supported_claims": 11,
    "time": 2.1
  },
  "test_case_mode_b_mistral_latest": {
    "coverage_score": 0.92,
    "total_claims": 12,
    "supported_claims": 11,
    "time": 4.2
  }
}
```

## Notes

- Experiments require Ollama to be running locally
- First run may be slower due to model loading
- Results may vary slightly between runs
- Ensure sufficient system resources for running LLM models

