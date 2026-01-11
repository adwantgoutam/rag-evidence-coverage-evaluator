"""Tests for evidence retrieval module.

Author: Goutam Adwant (gadwant)"""

import pytest
from ece.evidence_retriever import EvidenceRetriever
from ece.models import Passage, Claim, SupportingSnippet


class TestEvidenceRetriever:
    """Test cases for EvidenceRetriever."""
    
    @pytest.fixture
    def passages(self):
        """Create sample passages."""
        return [
            Passage(id="p1", text="The Eiffel Tower is located in Paris, France."),
            Passage(id="p2", text="It was completed in 1889 and stands 330 meters tall."),
            Passage(id="p3", text="The tower was designed by Gustave Eiffel."),
            Passage(id="p4", text="Paris is the capital city of France."),
        ]
    
    @pytest.fixture
    def claim(self):
        """Create a sample claim."""
        return Claim(
            text="The Eiffel Tower is in Paris",
            span=[0, 30],
            normalized="the eiffel tower is in paris"
        )
    
    def test_bm25_indexing(self, passages):
        """Test BM25 indexing."""
        retriever = EvidenceRetriever(method="bm25", top_k=2)
        retriever.index_passages(passages)
        
        assert len(retriever.passages) == 4
        assert hasattr(retriever, 'bm25')
    
    def test_bm25_retrieval(self, passages, claim):
        """Test BM25 retrieval."""
        retriever = EvidenceRetriever(method="bm25", top_k=2)
        retriever.index_passages(passages)
        
        snippets = retriever.retrieve(claim)
        
        assert len(snippets) > 0
        assert all(isinstance(s, SupportingSnippet) for s in snippets)
        # Should return passage about Eiffel Tower in Paris
        assert any("Paris" in s.text for s in snippets)
    
    def test_embedding_retrieval(self, passages, claim):
        """Test embedding-based retrieval."""
        retriever = EvidenceRetriever(method="embedding", top_k=2)
        retriever.index_passages(passages)
        
        snippets = retriever.retrieve(claim)
        
        assert len(snippets) > 0
        assert all(isinstance(s, SupportingSnippet) for s in snippets)
        assert all(s.score > 0 for s in snippets)
    
    def test_top_k_limit(self, passages, claim):
        """Test that top_k limit is respected."""
        retriever = EvidenceRetriever(method="bm25", top_k=2)
        retriever.index_passages(passages)
        
        snippets = retriever.retrieve(claim, top_k=2)
        assert len(snippets) <= 2
    
    def test_empty_passages(self):
        """Test handling of empty passages."""
        retriever = EvidenceRetriever(method="bm25")
        retriever.index_passages([])
        
        claim = Claim(text="Test", span=[0, 4])
        snippets = retriever.retrieve(claim)
        assert len(snippets) == 0
    
    def test_invalid_method(self):
        """Test error handling for invalid method."""
        with pytest.raises(ValueError):
            EvidenceRetriever(method="invalid")

