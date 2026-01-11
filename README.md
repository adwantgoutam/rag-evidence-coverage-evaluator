# Evidence Coverage Evaluator (ECE)

A comprehensive Python framework for evaluating how well RAG-generated answers are supported by retrieved evidence.

**Author**: gadwant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

The Evidence Coverage Evaluator (ECE) addresses a critical challenge in Retrieval-Augmented Generation (RAG) systems: evaluating whether generated answers are properly grounded in retrieved evidence. Unlike accuracy-based metrics, ECE measures *grounding* — whether each factual claim is explicitly supported by retrieved context.

### Key Features

- **Evidence Coverage Score**: Percentage of answer claims supported by retrieved context
- **Two Evaluation Modes**:
  - **Mode A (NLI-based)**: Fast, deterministic, no API costs (2-3 seconds)
  - **Mode B (LLM-based)**: Sophisticated reasoning with local Ollama models (23-31 seconds)
- **Fine-grained Claim Analysis**: Identifies unsupported claims with exact spans
- **Citation Quality Assessment**: Evaluates whether citations match supporting passages
- **Actionable Feedback**: Suggestions for improving evidence coverage
- **CI/CD Ready**: Designed for integration into continuous integration pipelines

### Framework Architecture

```
Answer Text → Claim Extraction → Evidence Retrieval → Entailment Scoring → Coverage Metrics
                    ↓                    ↓                    ↓
               SpaCy NLP           BM25/Embeddings      NLI Model (Mode A)
                                                        or LLM Judge (Mode B)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/gadwant/evidence-coverage-evaluator.git
cd evidence-coverage-evaluator

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package and dependencies
pip install -e .

# Download SpaCy English model (required)
python -m spacy download en_core_web_sm

# Verify installation
python -c "from ece import EvidenceCoverageEvaluator; print('Installation successful!')"
```

### Quick Install (Dependencies Only)

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Command Line Interface

```bash
# Basic evaluation
ece evaluate --answer examples/example_answer.txt --context examples/example_context.json

# With custom threshold and embedding retrieval
ece evaluate \
  --answer examples/example_answer.txt \
  --context examples/example_context.json \
  --threshold 0.8 \
  --retrieval-method embedding

# Generate HTML report
ece evaluate \
  --answer examples/example_answer.txt \
  --context examples/example_context.json \
  --output results.json \
  --html report.html
```

### Python API

```python
from ece import EvidenceCoverageEvaluator, Context, Passage

# Create context with passages
context = Context(passages=[
    Passage(id="p1", text="The Eiffel Tower is located in Paris, France."),
    Passage(id="p2", text="It was completed in 1889 and stands 330 meters tall."),
])

# Initialize evaluator
evaluator = EvidenceCoverageEvaluator(
    retrieval_method="bm25",  # or "embedding"
    threshold=0.7
)

# Evaluate answer
answer = "The Eiffel Tower is in Paris. It was built in 1889."
result = evaluator.evaluate(answer, context)

# Access results
print(f"Coverage Score: {result.coverage_score:.2%}")
print(f"Supported Claims: {result.supported_claims}/{result.total_claims}")
for claim in result.unsupported_claims:
    print(f"  Unsupported: {claim.claim}")
```

### Mode B with Ollama

For Mode B (LLM-based) evaluation, install Ollama and pull a model:

```bash
# Install Ollama (see https://ollama.ai)
ollama pull llama3:latest

# Use Mode B
from ece.ollama_judge import OllamaJudge
judge = OllamaJudge(model="llama3:latest")
```

## Input Format

### Answer File
Plain text file containing the generated answer.

### Context File (JSON)
```json
{
  "passages": [
    {"id": "passage_1", "text": "Passage content here..."},
    {"id": "passage_2", "text": "Another passage..."}
  ]
}
```

## Output Format

```json
{
  "coverage_score": 0.85,
  "total_claims": 10,
  "supported_claims": 8,
  "unsupported_claims": [
    {
      "claim": "Claim text...",
      "span": [0, 50],
      "missing_info": "Information about X is missing"
    }
  ],
  "claim_analysis": [...],
  "feedback": ["Retrieve more about X", ...]
}
```

## Project Structure

```
evidence-coverage-evaluator/
├── ece/                    # Core package
│   ├── __init__.py         # Package exports
│   ├── evaluator.py        # Main evaluation orchestrator
│   ├── claim_extractor.py  # Claim extraction module
│   ├── evidence_retriever.py # BM25/embedding retrieval
│   ├── nli_scorer.py       # NLI-based scoring (Mode A)
│   ├── ollama_judge.py     # LLM-based scoring (Mode B)
│   ├── citation_matcher.py # Citation quality analysis
│   ├── visualizer.py       # HTML report generation
│   ├── cli.py              # Command-line interface
│   └── models.py           # Data models
├── tests/                  # Test suite
├── examples/               # Example data files
├── documentation/          # Research paper and docs
└── requirements.txt        # Dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see [LICENSE](LICENSE) for details.
