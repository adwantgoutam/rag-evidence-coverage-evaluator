"""Tests for NLI scorer module.

Author: Goutam Adwant (gadwant)"""

import pytest
from ece.nli_scorer import NLIScorer
from ece.models import Claim, SupportingSnippet, ClaimAnalysis


class TestNLIScorer:
    """Test cases for NLIScorer."""
    
    @pytest.fixture
    def scorer(self):
        """Create NLIScorer with smaller model for testing."""
        # Use a smaller model for faster tests
        try:
            return NLIScorer(model_name="roberta-base-mnli", device="cpu")
        except Exception:
            pytest.skip("NLI model not available")
    
    @pytest.fixture
    def claim(self):
        """Create a sample claim."""
        return Claim(
            text="The Eiffel Tower is in Paris",
            span=[0, 30],
            normalized="the eiffel tower is in paris"
        )
    
    @pytest.fixture
    def supporting_snippet(self):
        """Create a supporting evidence snippet."""
        return SupportingSnippet(
            passage_id="p1",
            text="The Eiffel Tower is located in Paris, France.",
            score=0.0
        )
    
    @pytest.fixture
    def contradicting_snippet(self):
        """Create a contradicting evidence snippet."""
        return SupportingSnippet(
            passage_id="p2",
            text="The Eiffel Tower is located in London, England.",
            score=0.0
        )
    
    def test_entailment_scoring(self, scorer, claim, supporting_snippet):
        """Test entailment probability calculation."""
        prob = scorer._compute_entailment(
            supporting_snippet.text,
            claim.text
        )
        
        assert 0.0 <= prob <= 1.0
        # Should have high entailment for matching claim
        assert prob > 0.5
    
    def test_claim_scoring_with_support(self, scorer, claim, supporting_snippet):
        """Test scoring a claim with supporting evidence."""
        analysis = scorer.score_claim(claim, [supporting_snippet], threshold=0.5)
        
        assert isinstance(analysis, ClaimAnalysis)
        assert analysis.claim == claim
        assert analysis.support_score > 0
        assert len(analysis.supporting_snippets) > 0
    
    def test_claim_scoring_no_evidence(self, scorer, claim):
        """Test scoring a claim with no evidence."""
        analysis = scorer.score_claim(claim, [], threshold=0.5)
        
        assert not analysis.supported
        assert analysis.support_score == 0.0
        assert len(analysis.supporting_snippets) == 0
        assert analysis.missing_info is not None
    
    def test_threshold_filtering(self, scorer, claim, supporting_snippet):
        """Test that threshold filtering works."""
        # Low threshold
        analysis_low = scorer.score_claim(claim, [supporting_snippet], threshold=0.1)
        # High threshold
        analysis_high = scorer.score_claim(claim, [supporting_snippet], threshold=0.99)
        
        # Low threshold should include more snippets
        assert len(analysis_low.supporting_snippets) >= len(analysis_high.supporting_snippets)
    
    def test_best_snippet_selection(self, scorer, claim):
        """Test that best snippets are selected."""
        snippets = [
            SupportingSnippet(passage_id="p1", text="The Eiffel Tower is in Paris.", score=0.0),
            SupportingSnippet(passage_id="p2", text="Paris is a city in France.", score=0.0),
            SupportingSnippet(passage_id="p3", text="The weather is nice today.", score=0.0),
        ]
        
        analysis = scorer.score_claim(claim, snippets, threshold=0.3)
        
        # Should select best matching snippets
        assert len(analysis.supporting_snippets) > 0
        # Snippets should be sorted by score
        scores = [s.score for s in analysis.supporting_snippets]
        assert scores == sorted(scores, reverse=True)

