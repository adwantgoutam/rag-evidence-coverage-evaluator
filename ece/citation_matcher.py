"""Module for matching citations to supporting passages.

Author: Goutam Adwant (gadwant)

This module provides citation quality assessment functionality, including
extraction of various citation formats, matching citations to claims based
on proximity, and evaluating whether cited passages actually support claims.
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from ece.models import Passage, Claim, ClaimAnalysis, SupportingSnippet


class CitationMatcher:
    """Matches citations in answers to supporting passages.
    
    Analyzes citation quality by checking whether provided citations
    accurately reference the passages that actually support each claim.
    
    Attributes:
        citation_patterns: List of regex patterns for detecting citations
    """
    
    def __init__(self):
        """Initialize the citation matcher with common citation patterns."""
        self.citation_patterns = [
            r'\[(\d+)\]',       # [1], [2], etc.
            r'\((\d+)\)',       # (1), (2), etc.
            r'\[([A-Za-z]+)\]', # [A], [B], etc.
            r'\(([A-Za-z]+)\)', # (A), (B), etc.
            r'\[([A-Za-z]+\d+)\]',  # [A1], [B2], etc.
        ]
    
    def extract_citations(self, text: str) -> List[Tuple[str, int, int]]:
        """Extract citations from text.
        
        Detects various citation formats including numbered and lettered
        citations in brackets or parentheses.
        
        Args:
            text: Text containing citations
            
        Returns:
            List of (citation_id, start_pos, end_pos) tuples
        """
        citations = []
        
        for pattern in self.citation_patterns:
            for match in re.finditer(pattern, text):
                citation_id = match.group(1)
                start = match.start()
                end = match.end()
                citations.append((citation_id, start, end))
        
        # Remove duplicates and sort by position
        citations = list(set(citations))
        citations.sort(key=lambda x: x[1])
        
        return citations
    
    def find_citation_context(
        self,
        text: str,
        citation_pos: int,
        context_window: int = 100
    ) -> str:
        """Find the context around a citation.
        
        Args:
            text: Full text
            citation_pos: Position of citation
            context_window: Characters before and after citation
            
        Returns:
            Context string around citation
        """
        start = max(0, citation_pos - context_window)
        end = min(len(text), citation_pos + context_window)
        return text[start:end]
    
    def match_citations_to_claims(
        self,
        answer: str,
        claims: List[Claim],
        citations: List[Tuple[str, int, int]],
        passage_map: Dict[str, Passage]
    ) -> Dict[Claim, List[str]]:
        """Match citations to claims based on proximity.
        
        Associates each citation with the closest claim within a distance
        threshold, enabling evaluation of citation accuracy.
        
        Args:
            answer: Full answer text
            claims: List of claims
            citations: List of (citation_id, start, end) tuples
            passage_map: Dictionary mapping citation IDs to passages
            
        Returns:
            Dictionary mapping claims to list of citation IDs
        """
        claim_citations = {claim: [] for claim in claims}
        
        for citation_id, cit_start, cit_end in citations:
            closest_claim = None
            min_distance = float('inf')
            
            for claim in claims:
                claim_start, claim_end = claim.span
                
                # Calculate distance (overlap or proximity)
                if cit_start <= claim_end and cit_end >= claim_start:
                    distance = 0  # Citation overlaps with claim
                else:
                    distance = min(
                        abs(cit_start - claim_end),
                        abs(cit_end - claim_start)
                    )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_claim = claim
            
            # Associate if within 200 character distance
            if closest_claim and min_distance < 200:
                if citation_id in passage_map:
                    claim_citations[closest_claim].append(citation_id)
        
        return claim_citations
    
    def evaluate_citation_quality(
        self,
        claim: Claim,
        claim_analysis: ClaimAnalysis,
        associated_citations: List[str],
        passage_map: Dict[str, Passage]
    ) -> Dict[str, Any]:
        """Evaluate if citations match the supporting passages.
        
        Checks whether the cited passages are the same ones identified
        as actually supporting the claim through NLI scoring.
        
        Args:
            claim: The claim being evaluated
            claim_analysis: Analysis of the claim
            associated_citations: List of citation IDs associated with the claim
            passage_map: Dictionary mapping citation IDs to passages
            
        Returns:
            Dictionary with citation quality metrics
        """
        result = {
            "has_citations": len(associated_citations) > 0,
            "citation_count": len(associated_citations),
            "matching_citations": [],
            "mismatched_citations": [],
            "missing_citations": [],
            "citation_quality_score": 0.0
        }
        
        if not claim_analysis.supported:
            result["citation_quality_score"] = 0.0
            return result
        
        # Get passage IDs from supporting snippets
        supporting_passage_ids = {
            snippet.passage_id for snippet in claim_analysis.supporting_snippets
        }
        
        # Check each citation
        for citation_id in associated_citations:
            if citation_id in passage_map:
                cited_passage = passage_map[citation_id]
                
                if cited_passage.id in supporting_passage_ids:
                    result["matching_citations"].append(citation_id)
                else:
                    result["mismatched_citations"].append(citation_id)
        
        # Find supporting passages that weren't cited
        for snippet in claim_analysis.supporting_snippets:
            cited_passage_ids = [
                passage_map.get(c, Passage(id="", text="")).id 
                for c in associated_citations
            ]
            if snippet.passage_id not in cited_passage_ids:
                result["missing_citations"].append(snippet.passage_id)
        
        # Calculate quality score
        if len(associated_citations) > 0:
            matching_ratio = len(result["matching_citations"]) / len(associated_citations)
            result["citation_quality_score"] = matching_ratio
        elif len(supporting_passage_ids) > 0:
            result["citation_quality_score"] = 0.0  # Should have citations
        else:
            result["citation_quality_score"] = 1.0  # No citations needed
        
        return result
    
    def analyze_citations(
        self,
        answer: str,
        claims: List[Claim],
        claim_analyses: List[ClaimAnalysis],
        passages: List[Passage],
        passage_id_to_citation: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Complete citation analysis.
        
        Performs end-to-end citation quality assessment including extraction,
        matching, and quality scoring for all claims.
        
        Args:
            answer: Full answer text
            claims: List of claims
            claim_analyses: List of claim analyses
            passages: List of passages
            passage_id_to_citation: Optional mapping from passage IDs to citation IDs
            
        Returns:
            Dictionary with citation analysis results
        """
        # Extract citations from answer
        citations = self.extract_citations(answer)
        
        # Create passage map
        passage_map = {}
        if passage_id_to_citation:
            for passage in passages:
                if passage.id in passage_id_to_citation:
                    citation_id = passage_id_to_citation[passage.id]
                    passage_map[citation_id] = passage
        else:
            # Default: use passage IDs as citation IDs
            for passage in passages:
                passage_map[passage.id] = passage
        
        # Match citations to claims
        claim_to_citations = self.match_citations_to_claims(
            answer, claims, citations, passage_map
        )
        
        # Evaluate citation quality for each claim
        citation_analyses = []
        overall_quality_scores = []
        
        for claim, analysis in zip(claims, claim_analyses):
            associated_citations = claim_to_citations.get(claim, [])
            quality = self.evaluate_citation_quality(
                claim, analysis, associated_citations, passage_map
            )
            citation_analyses.append({
                "claim": claim.text,
                "quality": quality
            })
            overall_quality_scores.append(quality["citation_quality_score"])
        
        # Calculate overall citation quality
        overall_quality = (
            sum(overall_quality_scores) / len(overall_quality_scores)
            if overall_quality_scores else 0.0
        )
        
        return {
            "total_citations": len(citations),
            "citation_analyses": citation_analyses,
            "overall_citation_quality": overall_quality,
            "citation_spam_score": self._calculate_spam_score(citations, claims)
        }
    
    def _calculate_spam_score(
        self,
        citations: List[Tuple[str, int, int]],
        claims: List[Claim]
    ) -> float:
        """Calculate citation spam score.
        
        Detects excessive or irrelevant citations that may indicate
        citation spam rather than meaningful attribution.
        
        Args:
            citations: List of citations
            claims: List of claims
            
        Returns:
            Spam score (0-1, higher means more spam)
        """
        if len(claims) == 0:
            return 1.0 if len(citations) > 0 else 0.0
        
        citations_per_claim = len(citations) / len(claims)
        
        # More than 3 citations per claim is considered spam
        if citations_per_claim > 3:
            return min(1.0, (citations_per_claim - 3) / 3)
        
        return 0.0
