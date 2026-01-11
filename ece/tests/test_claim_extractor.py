"""Tests for claim extraction module.

Author: Goutam Adwant (gadwant)"""

import pytest
from ece.claim_extractor import ClaimExtractor
from ece.models import Claim


class TestClaimExtractor:
    """Test cases for ClaimExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create a ClaimExtractor instance."""
        return ClaimExtractor()
    
    def test_basic_sentence_extraction(self, extractor):
        """Test basic sentence extraction."""
        answer = "The Eiffel Tower is in Paris. It was built in 1889."
        claims = extractor.extract_claims(answer)
        
        assert len(claims) >= 2
        assert all(isinstance(c, Claim) for c in claims)
        assert "Eiffel Tower" in claims[0].text
    
    def test_conjunction_splitting(self, extractor):
        """Test splitting on conjunctions."""
        answer = "The tower is tall and it is located in Paris."
        claims = extractor.extract_claims(answer)
        
        # Should split on "and"
        assert len(claims) >= 2
    
    def test_enumeration_handling(self, extractor):
        """Test handling of enumerations."""
        answer = "1. First point. 2. Second point. 3. Third point."
        claims = extractor.extract_claims(answer)
        
        # Should handle enumerations
        assert len(claims) >= 3
    
    def test_normalization(self, extractor):
        """Test claim normalization."""
        answer = "The tower is 1,000 meters tall."
        claims = extractor.extract_claims(answer)
        
        assert len(claims) > 0
        # Check normalization removes commas from numbers
        if claims[0].normalized:
            assert "1,000" not in claims[0].normalized or "1000" in claims[0].normalized
    
    def test_empty_answer(self, extractor):
        """Test handling of empty answer."""
        claims = extractor.extract_claims("")
        assert len(claims) == 0
    
    def test_span_calculation(self, extractor):
        """Test that spans are correctly calculated."""
        answer = "First sentence. Second sentence."
        claims = extractor.extract_claims(answer)
        
        assert len(claims) >= 2
        # Spans should be valid
        for claim in claims:
            assert len(claim.span) == 2
            assert claim.span[0] < claim.span[1]
            assert claim.span[0] >= 0
            assert claim.span[1] <= len(answer)

