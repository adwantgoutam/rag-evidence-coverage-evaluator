"""Data models for the Evidence Coverage Evaluator.

Author: Goutam Adwant (gadwant)

This module defines the core data structures used throughout the ECE framework,
including passages, claims, and evaluation results. All models use Pydantic
for type validation and serialization support.
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field


class Passage(BaseModel):
    """A retrieved passage with ID and text.
    
    Attributes:
        id: Unique identifier for the passage
        text: The passage content
    """
    id: str
    text: str


class Context(BaseModel):
    """Retrieved context containing multiple passages.
    
    Attributes:
        passages: List of Passage objects representing the retrieved context
    """
    passages: List[Passage]


class SupportingSnippet(BaseModel):
    """A snippet from a passage that supports a claim.
    
    Attributes:
        passage_id: ID of the source passage
        text: The snippet text
        score: Support score between 0 and 1
    """
    passage_id: str
    text: str
    score: float = Field(description="Support score (0-1)")


class Claim(BaseModel):
    """An extracted claim from the answer.
    
    Attributes:
        text: The original claim text
        span: Character span [start, end] in the original answer
        normalized: Normalized version of the claim for improved matching
    """
    text: str
    span: Tuple[int, int] = Field(description="Character span [start, end] in original answer")
    normalized: Optional[str] = Field(None, description="Normalized version of the claim")
    
    model_config = {"frozen": True}


class ClaimAnalysis(BaseModel):
    """Analysis of a single claim's support status.
    
    Attributes:
        claim: The claim being analyzed
        supported: Whether the claim is supported by evidence
        support_score: Maximum entailment score across all evidence
        supporting_snippets: List of snippets that support the claim
        missing_info: Description of what information is missing if not supported
    """
    claim: Claim
    supported: bool
    support_score: float = Field(description="Support score (0-1)")
    supporting_snippets: List[SupportingSnippet] = []
    missing_info: Optional[str] = Field(None, description="What information is missing if not supported")


class UnsupportedClaim(BaseModel):
    """A claim that lacks sufficient evidence.
    
    Attributes:
        claim: The unsupported claim text
        span: Character span in the original answer
        missing_info: Description of missing evidence
    """
    claim: str
    span: List[int]
    missing_info: Optional[str] = None


class EvaluationResult(BaseModel):
    """Complete evaluation result from the ECE framework.
    
    Attributes:
        coverage_score: Proportion of claims supported by evidence (0-1)
        total_claims: Total number of extracted claims
        supported_claims: Number of claims with sufficient evidence
        unsupported_claims: List of claims lacking evidence
        claim_analysis: Detailed analysis for each claim
        feedback: Actionable suggestions for improvement
        metadata: Additional evaluation metadata
    """
    coverage_score: float = Field(description="Overall coverage score (0-1)")
    total_claims: int
    supported_claims: int
    unsupported_claims: List[UnsupportedClaim] = []
    claim_analysis: List[ClaimAnalysis] = []
    feedback: List[str] = Field(default_factory=list, description="Actionable feedback")
    metadata: Dict[str, Any] = Field(default_factory=dict)
