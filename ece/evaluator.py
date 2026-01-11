"""Main evaluation orchestrator for Evidence Coverage Evaluator.

Author: Goutam Adwant (gadwant)

This module provides the EvidenceCoverageEvaluator class, which orchestrates
the complete evaluation pipeline: claim extraction, evidence retrieval,
entailment scoring, and feedback generation.
"""

from typing import List
from ece.models import (
    Context, Claim, EvaluationResult, ClaimAnalysis, UnsupportedClaim
)
from ece.claim_extractor import ClaimExtractor
from ece.evidence_retriever import EvidenceRetriever
from ece.nli_scorer import NLIScorer
from ece.citation_matcher import CitationMatcher


class EvidenceCoverageEvaluator:
    """Main evaluator that orchestrates the evidence coverage evaluation pipeline.
    
    This class coordinates the complete evaluation workflow including claim
    extraction, evidence retrieval, NLI-based entailment scoring, and
    citation quality assessment.
    
    Attributes:
        threshold: Minimum support score to consider a claim supported
        claim_extractor: ClaimExtractor instance for decomposing answers
        evidence_retriever: EvidenceRetriever instance for finding relevant passages
        citation_matcher: CitationMatcher instance for citation quality analysis
        scorer: NLIScorer instance for entailment scoring
    """
    
    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_top_k: int = 3,
        nli_model: str = "roberta-large-mnli",
        threshold: float = 0.7
    ):
        """Initialize the evaluator with configurable components.
        
        Args:
            retrieval_method: Evidence retrieval method - "bm25" or "embedding"
            retrieval_top_k: Number of evidence candidates to retrieve per claim
            nli_model: HuggingFace NLI model name for entailment scoring
            threshold: Minimum support score to consider a claim supported (0-1)
        """
        self.threshold = threshold
        
        # Initialize pipeline components
        self.claim_extractor = ClaimExtractor()
        self.evidence_retriever = EvidenceRetriever(
            method=retrieval_method,
            top_k=retrieval_top_k
        )
        self.citation_matcher = CitationMatcher()
        self.scorer = NLIScorer(model_name=nli_model)
    
    def evaluate(
        self,
        answer: str,
        context: Context
    ) -> EvaluationResult:
        """Evaluate evidence coverage for a RAG-generated answer.
        
        This method executes the complete evaluation pipeline:
        1. Extract claims from the answer
        2. Index context passages for retrieval
        3. Retrieve relevant evidence for each claim
        4. Score claim-evidence pairs using NLI
        5. Calculate coverage metrics
        6. Analyze citations if present
        7. Generate actionable feedback
        
        Args:
            answer: The generated answer text to evaluate
            context: Retrieved context containing source passages
            
        Returns:
            EvaluationResult with coverage score, claim analyses, and feedback
        """
        # Step 1: Extract claims from the answer
        print("Extracting claims from answer...")
        claims = self.claim_extractor.extract_claims(answer)
        print(f"Extracted {len(claims)} claims")
        
        # Step 2: Index passages for retrieval
        print("Indexing passages for retrieval...")
        self.evidence_retriever.index_passages(context.passages)
        
        # Step 3: Retrieve evidence for each claim
        print("Retrieving evidence for each claim...")
        evidence_dict = {}
        for claim in claims:
            evidence = self.evidence_retriever.retrieve(claim)
            evidence_dict[claim] = evidence
        
        # Step 4: Score claims using NLI model
        print("Scoring claims using NLI model...")
        claim_analyses = []
        for claim in claims:
            evidence = evidence_dict[claim]
            analysis = self.scorer.score_claim(claim, evidence, self.threshold)
            claim_analyses.append(analysis)
        
        # Step 5: Calculate coverage metrics
        supported_count = sum(1 for a in claim_analyses if a.supported)
        coverage_score = supported_count / len(claims) if claims else 0.0
        
        # Step 6: Identify unsupported claims
        unsupported_claims = []
        for analysis in claim_analyses:
            if not analysis.supported:
                unsupported_claims.append(UnsupportedClaim(
                    claim=analysis.claim.text,
                    span=analysis.claim.span,
                    missing_info=analysis.missing_info
                ))
        
        # Step 7: Generate actionable feedback
        feedback = self._generate_feedback(claim_analyses, unsupported_claims)
        
        # Step 8: Analyze citations if present in the answer
        citation_analysis = None
        citations = self.citation_matcher.extract_citations(answer)
        if citations:
            print("Analyzing citations...")
            citation_analysis = self.citation_matcher.analyze_citations(
                answer, claims, claim_analyses, context.passages
            )
        
        # Build metadata
        metadata = {
            "mode": "lightweight",
            "threshold": self.threshold,
            "retrieval_method": self.evidence_retriever.method
        }
        
        if citation_analysis:
            metadata["citation_analysis"] = citation_analysis
        
        return EvaluationResult(
            coverage_score=coverage_score,
            total_claims=len(claims),
            supported_claims=supported_count,
            unsupported_claims=unsupported_claims,
            claim_analysis=claim_analyses,
            feedback=feedback,
            metadata=metadata
        )
    
    def _generate_feedback(
        self, 
        analyses: List[ClaimAnalysis], 
        unsupported: List[UnsupportedClaim]
    ) -> List[str]:
        """Generate actionable feedback based on evaluation results.
        
        Provides topic-level and claim-level suggestions for improving
        evidence coverage in RAG systems.
        
        Args:
            analyses: List of claim analysis results
            unsupported: List of unsupported claims
            
        Returns:
            List of feedback strings with improvement suggestions
        """
        feedback = []
        
        if not unsupported:
            feedback.append("All claims are supported by evidence. Great coverage!")
            return feedback
        
        # Extract topics from unsupported claims for retrieval suggestions
        topics = set()
        for claim in unsupported:
            words = claim.claim.lower().split()
            # Filter out common words to extract key terms
            common_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being"}
            key_words = [w for w in words if w not in common_words and len(w) > 3]
            if key_words:
                topics.add(key_words[0])
        
        # Generate topic-level feedback
        for topic in topics:
            feedback.append(f"Retrieve more information about: {topic}")
        
        # Generate specific claim feedback (limit to top 3)
        for claim in unsupported[:3]:
            if claim.missing_info:
                feedback.append(f"Unsupported claim: '{claim.claim[:50]}...' - {claim.missing_info}")
        
        return feedback
