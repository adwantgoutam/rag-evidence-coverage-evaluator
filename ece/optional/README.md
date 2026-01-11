# Optional Modules

This folder contains optional functionality that is not part of the core Evidence Coverage Evaluator (Mode A).

## Mode B: LLM-Judge

The `llm_judge.py` module provides LLM-based claim evaluation functionality. This is kept separate from the core package as it requires additional dependencies and API keys.

### Installation

To use Mode B (LLM-Judge), install the required dependencies:

```bash
# For OpenAI
pip install openai>=1.0.0

# For Anthropic
pip install anthropic>=0.7.0
```

### Usage

This module is not currently integrated into the main CLI. It is kept here for future use and can be imported directly if needed:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from optional.llm_judge import LLMJudge
from ece.models import Claim, SupportingSnippet

# Initialize judge
judge = LLMJudge(
    provider="openai",
    model="gpt-4",
    api_key="your-api-key"
)

# Use it to score claims
# (See llm_judge.py for full API documentation)
```

### Note

This functionality will be integrated into the main package in a future release after further testing and validation.

