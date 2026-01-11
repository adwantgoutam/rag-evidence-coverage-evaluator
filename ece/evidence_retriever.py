"""Module for retrieving evidence from context passages.

Author: Goutam Adwant (gadwant)

This module provides evidence retrieval using either BM25 (sparse retrieval)
or sentence-transformer embeddings (dense retrieval). Retrieved passages are
ranked by relevance score and returned as SupportingSnippet objects.
"""

from typing import List, Optional
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from ece.models import Passage, Claim, SupportingSnippet


class EvidenceRetriever:
    """Retrieves evidence snippets from passages for given claims.
    
    Supports two retrieval methods:
    - BM25: Sparse lexical matching using the Okapi BM25 ranking function
    - Embedding: Dense semantic matching using sentence transformers
    
    Attributes:
        method: Retrieval method in use ("bm25" or "embedding")
        top_k: Number of evidence candidates to return per claim
        passages: Indexed passages for retrieval
        passage_texts: Text content of indexed passages
    """
    
    def __init__(
        self,
        method: str = "bm25",
        embedding_model: Optional[str] = None,
        top_k: int = 3
    ):
        """Initialize the evidence retriever.
        
        Args:
            method: Retrieval method - "bm25" for sparse or "embedding" for dense
            embedding_model: Model name for embedding-based retrieval.
                           Defaults to "all-MiniLM-L6-v2" if method is "embedding"
            top_k: Number of top evidence candidates to return per claim
            
        Raises:
            ValueError: If an unknown retrieval method is specified
        """
        self.method = method
        self.top_k = top_k
        self.embedding_model = None
        self.passages = []
        self.passage_texts = []
        
        if method == "embedding":
            model_name = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
            self.embedding_model = SentenceTransformer(model_name)
        elif method not in ["bm25", "embedding"]:
            raise ValueError(f"Unknown method: {method}. Use 'bm25' or 'embedding'")
    
    def index_passages(self, passages: List[Passage]):
        """Index passages for retrieval.
        
        Pre-computes BM25 index or passage embeddings depending on the
        configured retrieval method.
        
        Args:
            passages: List of Passage objects to index
        """
        self.passages = passages
        self.passage_texts = [p.text for p in passages]
        
        if not passages:
            return
            
        if self.method == "bm25":
            try:
                # Simple tokenization by splitting on whitespace
                tokenized_passages = [p.text.lower().split() for p in passages]
                self.bm25 = BM25Okapi(tokenized_passages)
            except Exception as e:
                print(f"Error initializing BM25: {e}")
                self.bm25 = None
        elif self.method == "embedding":
            # Pre-compute embeddings for all passages
            self.passage_embeddings = self.embedding_model.encode(
                self.passage_texts,
                show_progress_bar=False
            )
    
    def retrieve(self, claim: Claim, top_k: Optional[int] = None) -> List[SupportingSnippet]:
        """Retrieve top-k evidence snippets for a claim.
        
        Args:
            claim: The claim to find evidence for
            top_k: Number of snippets to return (overrides initialization value)
            
        Returns:
            List of SupportingSnippet objects sorted by relevance score (descending)
        """
        if not self.passages:
            return []
        
        k = top_k or self.top_k
        
        if self.method == "bm25":
            return self._retrieve_bm25(claim, k)
        else:
            return self._retrieve_embedding(claim, k)
    
    def _retrieve_bm25(self, claim: Claim, k: int) -> List[SupportingSnippet]:
        """Retrieve using BM25 sparse matching.
        
        Args:
            claim: Claim to retrieve evidence for
            k: Number of passages to retrieve
            
        Returns:
            List of SupportingSnippet objects
        """
        query_text = claim.normalized or claim.text
        tokenized_query = self._tokenize(query_text)
        
        # Get BM25 scores for all passages
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices by score
        top_indices = np.argsort(scores)[::-1][:k]
        
        snippets = []
        for idx in top_indices:
            if scores[idx] > 0:
                snippets.append(SupportingSnippet(
                    passage_id=self.passages[idx].id,
                    text=self.passage_texts[idx],
                    score=float(scores[idx])
                ))
        
        return snippets
    
    def _retrieve_embedding(self, claim: Claim, k: int) -> List[SupportingSnippet]:
        """Retrieve using dense embedding similarity.
        
        Args:
            claim: Claim to retrieve evidence for
            k: Number of passages to retrieve
            
        Returns:
            List of SupportingSnippet objects
        """
        query_text = claim.normalized or claim.text
        
        # Encode the claim
        query_embedding = self.embedding_model.encode([query_text], show_progress_bar=False)[0]
        
        # Compute cosine similarity with all passages
        similarities = np.dot(self.passage_embeddings, query_embedding) / (
            np.linalg.norm(self.passage_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k indices by similarity
        top_indices = np.argsort(similarities)[::-1][:k]
        
        snippets = []
        for idx in top_indices:
            if similarities[idx] > 0:
                snippets.append(SupportingSnippet(
                    passage_id=self.passages[idx].id,
                    text=self.passage_texts[idx],
                    score=float(similarities[idx])
                ))
        
        return snippets
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25.
        
        Converts text to lowercase and splits on word boundaries.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of lowercase tokens
        """
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
