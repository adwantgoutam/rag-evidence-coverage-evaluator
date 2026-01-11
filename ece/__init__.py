"""Evidence Coverage Evaluator - A comprehensive framework for assessing
grounding quality in Retrieval-Augmented Generation (RAG) systems.

Author: Goutam Adwant (gadwant)
Repository: https://github.com/gadwant/evidence-coverage-evaluator
License: MIT

This package provides tools for evaluating whether RAG-generated answers
are properly grounded in retrieved evidence, supporting both lightweight
NLI-based evaluation (Mode A) and LLM-based judge evaluation (Mode B).
"""

__version__ = "0.1.0"
__author__ = "Goutam Adwant"
__email__ = "workwithgoutam@gmail.com"

# Core data model exports (no heavy dependencies)
from ece.models import (
    Passage,
    Context,
    Claim,
    ClaimAnalysis,
    SupportingSnippet,
    UnsupportedClaim,
    EvaluationResult
)

# Lazy imports for components with heavy dependencies
def get_evaluator():
    """Get the EvidenceCoverageEvaluator class (lazy import)."""
    from ece.evaluator import EvidenceCoverageEvaluator
    return EvidenceCoverageEvaluator

# For backward compatibility, import evaluator when module is used
# This allows: from ece import EvidenceCoverageEvaluator
def __getattr__(name):
    if name == "EvidenceCoverageEvaluator":
        from ece.evaluator import EvidenceCoverageEvaluator
        return EvidenceCoverageEvaluator
    raise AttributeError(f"module 'ece' has no attribute '{name}'")

__all__ = [
    "EvidenceCoverageEvaluator",
    "Passage",
    "Context", 
    "Claim",
    "ClaimAnalysis",
    "SupportingSnippet",
    "UnsupportedClaim",
    "EvaluationResult",
    "get_evaluator",
]
