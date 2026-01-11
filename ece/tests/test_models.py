"""Tests for data models.

Author: Goutam Adwant (gadwant)"""

import pytest
from ece.models import (
    Passage, Context, Claim, SupportingSnippet,
    ClaimAnalysis, UnsupportedClaim, EvaluationResult
)


class TestModels:
    """Test cases for data models."""
    
    def test_passage_model(self):
        """Test Passage model."""
        passage = Passage(id="test", text="Test passage")
        assert passage.id == "test"
        assert passage.text == "Test passage"
    
    def test_context_model(self):
        """Test Context model."""
        passages = [
            Passage(id="p1", text="Text 1"),
            Passage(id="p2", text="Text 2"),
        ]
        context = Context(passages=passages)
        assert len(context.passages) == 2
    
    def test_claim_model(self):
        """Test Claim model."""
        claim = Claim(
            text="Test claim",
            span=[0, 10],
            normalized="test claim"
        )
        assert claim.text == "Test claim"
        assert claim.span == (0, 10)
        assert claim.normalized == "test claim"
    
    def test_supporting_snippet_model(self):
        """Test SupportingSnippet model."""
        snippet = SupportingSnippet(
            passage_id="p1",
            text="Supporting text",
            score=0.85
        )
        assert snippet.passage_id == "p1"
        assert snippet.text == "Supporting text"
        assert snippet.score == 0.85
    
    def test_claim_analysis_model(self):
        """Test ClaimAnalysis model."""
        claim = Claim(text="Test", span=[0, 4])
        analysis = ClaimAnalysis(
            claim=claim,
            supported=True,
            support_score=0.9,
            supporting_snippets=[]
        )
        assert analysis.claim == claim
        assert analysis.supported is True
        assert analysis.support_score == 0.9
    
    def test_evaluation_result_model(self):
        """Test EvaluationResult model."""
        result = EvaluationResult(
            coverage_score=0.85,
            total_claims=10,
            supported_claims=8,
            unsupported_claims=[],
            claim_analysis=[],
            feedback=[]
        )
        assert result.coverage_score == 0.85
        assert result.total_claims == 10
        assert result.supported_claims == 8

