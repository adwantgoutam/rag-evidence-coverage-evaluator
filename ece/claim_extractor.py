"""Module for extracting and normalizing claims from answers.

Author: Goutam Adwant (gadwant)

This module provides claim extraction functionality using SpaCy for sentence
segmentation combined with heuristic rules for sub-claim splitting. Claims
are normalized for improved matching during evidence retrieval.
"""

import re
from typing import List, Tuple
import spacy
from ece.models import Claim


class ClaimExtractor:
    """Extracts claims from answers using sentence segmentation and heuristics.
    
    This class decomposes RAG-generated answers into atomic, verifiable claims
    that can be individually evaluated for evidence support.
    
    Attributes:
        nlp: SpaCy language model for sentence segmentation
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the claim extractor with a SpaCy model.
        
        Args:
            model_name: SpaCy model name for sentence segmentation.
                       Defaults to "en_core_web_sm".
                       
        Raises:
            OSError: If the specified SpaCy model is not installed
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            raise OSError(
                f"SpaCy model '{model_name}' not found. "
                f"Install it with: python -m spacy download {model_name}"
            )
    
    def extract_claims(self, answer: str) -> List[Claim]:
        """Extract atomic claims from an answer text.
        
        Decomposes the answer into sentences, then further splits sentences
        into sub-claims using heuristic rules for coordinating conjunctions
        and comma-separated clauses.
        
        Args:
            answer: The generated answer text to decompose
            
        Returns:
            List of Claim objects with text, spans, and normalized versions
        """
        claims = []
        current_pos = 0
        
        # Split into sentences using SpaCy
        doc = self.nlp(answer)
        sentences = [sent.text for sent in doc.sents]
        
        for sentence in sentences:
            sentence_start = answer.find(sentence, current_pos)
            if sentence_start == -1:
                sentence_start = current_pos
            
            # Further split sentence using heuristics
            sub_claims = self._split_sentence(sentence)
            
            for sub_claim in sub_claims:
                claim_text = sub_claim.strip()
                if not claim_text:
                    continue
                
                # Find position in original answer
                claim_start = answer.find(claim_text, sentence_start)
                if claim_start == -1:
                    claim_start = sentence_start
                
                claim_end = claim_start + len(claim_text)
                
                # Normalize the claim for improved matching
                normalized = self._normalize_claim(claim_text)
                
                claims.append(Claim(
                    text=claim_text,
                    span=[claim_start, claim_end],
                    normalized=normalized
                ))
            
            current_pos = sentence_start + len(sentence)
        
        return claims
    
    def _split_sentence(self, sentence: str) -> List[str]:
        """Split a sentence into sub-claims using heuristics.
        
        Heuristics applied:
        - Split on coordinating conjunctions (and, but, or, nor, yet, so)
        - Split on commas separating distinct propositions
        - Preserve enumeration structures
        
        Args:
            sentence: A single sentence to split
            
        Returns:
            List of sub-claim strings
        """
        sentence = sentence.strip()
        if not sentence:
            return []
        
        # Pattern for coordinating conjunctions
        conjunctions = r'\s+(and|but|or|nor|yet|so)\s+'
        parts = re.split(conjunctions, sentence)
        
        if len(parts) == 1:
            # No conjunctions found, try comma-based splitting
            return self._split_on_commas_and_enums(sentence)
        
        # Reconstruct parts (alternating [text, conjunction, text, ...])
        sub_claims = []
        current = parts[0]
        
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                sub_claims.append(current.strip())
                current = parts[i] + " " + parts[i + 1]
            else:
                current += " " + parts[i]
        
        if current.strip():
            sub_claims.append(current.strip())
        
        if len(sub_claims) > 1:
            return sub_claims
        else:
            return self._split_on_commas_and_enums(sentence)
    
    def _split_on_commas_and_enums(self, sentence: str) -> List[str]:
        """Split on commas and enumerations.
        
        Splits on commas followed by space, but preserves numbers
        with embedded commas (e.g., "1,000").
        
        Args:
            sentence: Sentence to split
            
        Returns:
            List of sub-claim strings
        """
        # Split on commas but not in numbers like 1,000
        parts = re.split(r',\s+(?![0-9])', sentence)
        
        # Check for enumeration patterns (1., 2., a., b., etc.)
        enum_pattern = r'^\s*([0-9]+|[a-z])\.\s+'
        if re.match(enum_pattern, sentence):
            return [sentence]
        
        return [p.strip() for p in parts if p.strip()]
    
    def _normalize_claim(self, claim: str) -> str:
        """Normalize a claim for improved matching.
        
        Normalizations applied:
        - Convert to lowercase
        - Remove commas from numbers (e.g., "1,000" -> "1000")
        - Expand common unit abbreviations
        - Remove extra whitespace
        
        Args:
            claim: Raw claim text
            
        Returns:
            Normalized claim string
        """
        normalized = claim.lower()
        
        # Normalize numbers: remove commas in numbers
        normalized = re.sub(r'(\d),(\d)', r'\1\2', normalized)
        
        # Normalize common units
        unit_mappings = {
            r'\bkm\b': 'kilometer',
            r'\bmi\b': 'mile',
            r'\bkg\b': 'kilogram',
            r'\blb\b': 'pound',
            r'\bhr\b': 'hour',
            r'\bmin\b': 'minute',
            r'\bsec\b': 'second',
        }
        
        for pattern, replacement in unit_mappings.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
