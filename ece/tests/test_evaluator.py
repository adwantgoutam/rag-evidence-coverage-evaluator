"""Tests for main evaluator module.

Author: Goutam Adwant (gadwant)"""

import pytest
from ece.evaluator import EvidenceCoverageEvaluator
from ece.models import Context, Passage


class TestEvidenceCoverageEvaluator:
    """Test cases for EvidenceCoverageEvaluator."""
    
    @pytest.fixture
    def context(self):
        """Create sample context."""
        return Context(
            passages=[
                Passage(id="p1", text="The Eiffel Tower is located in Paris, France."),
                Passage(id="p2", text="It was completed in 1889 and stands 330 meters tall."),
                Passage(id="p3", text="The tower was designed by Gustave Eiffel."),
            ]
        )
    
    @pytest.fixture
    def answer(self):
        """Create sample answer."""
        return "The Eiffel Tower is in Paris. It was built in 1889 and is 330 meters tall."
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = EvidenceCoverageEvaluator()
        
        assert evaluator.scorer is not None
        assert evaluator.threshold == 0.7
    
    def test_evaluation_pipeline(self, context, answer):
        """Test the full evaluation pipeline."""
        evaluator = EvidenceCoverageEvaluator(
            threshold=0.5
        )
        
        result = evaluator.evaluate(answer, context)
        
        assert result.coverage_score >= 0.0
        assert result.coverage_score <= 1.0
        assert result.total_claims > 0
        assert result.supported_claims >= 0
        assert result.supported_claims <= result.total_claims
        assert len(result.claim_analysis) == result.total_claims
        assert isinstance(result.feedback, list)
    
    def test_empty_answer(self, context):
        """Test handling of empty answer."""
        evaluator = EvidenceCoverageEvaluator()
        result = evaluator.evaluate("", context)
        
        assert result.total_claims == 0
        assert result.coverage_score == 0.0
    
    def test_empty_context(self, answer):
        """Test handling of empty context."""
        empty_context = Context(passages=[])
        evaluator = EvidenceCoverageEvaluator()
        result = evaluator.evaluate(answer, empty_context)
        
        assert result.total_claims > 0
        assert result.supported_claims == 0
        assert result.coverage_score == 0.0

