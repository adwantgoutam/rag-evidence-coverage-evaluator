"""Module for NLI-based entailment scoring (Mode A).

Author: Goutam Adwant (gadwant)

This module implements Mode A evaluation using Natural Language Inference (NLI)
models to assess entailment between claims and evidence. The NLI model treats
evidence as premises and claims as hypotheses, returning entailment probabilities.
"""

from typing import List, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ece.models import Claim, SupportingSnippet, ClaimAnalysis


class NLIScorer:
    """Scores claim-evidence pairs using Natural Language Inference.
    
    Uses pre-trained NLI models (e.g., RoBERTa-MNLI) to compute entailment
    probabilities between evidence passages (premises) and claims (hypotheses).
    
    Attributes:
        model_name: Name of the HuggingFace NLI model
        device: Compute device (cuda or cpu)
        batch_size: Batch size for inference
        tokenizer: Model tokenizer
        model: NLI classification model
        entailment_label_id: Index of the entailment label (typically 2)
    """
    
    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        device: Optional[str] = None,
        batch_size: int = 8
    ):
        """Initialize the NLI scorer.
        
        Args:
            model_name: HuggingFace model name for NLI.
                       Options: "roberta-large-mnli", "facebook/bart-large-mnli", etc.
            device: Device to run on ("cuda", "cpu", or None for auto-detection)
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        print(f"Loading NLI model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # MNLI models have 3 labels: contradiction (0), neutral (1), entailment (2)
        self.entailment_label_id = 2
    
    def score_claim(
        self,
        claim: Claim,
        evidence_snippets: List[SupportingSnippet],
        threshold: float = 0.7
    ) -> ClaimAnalysis:
        """Score a claim against evidence snippets using NLI.
        
        Computes entailment probability for each claim-evidence pair and
        determines if the claim is supported based on the threshold.
        
        Args:
            claim: The claim to evaluate
            evidence_snippets: List of evidence snippets to check against
            threshold: Minimum entailment probability to consider supported
            
        Returns:
            ClaimAnalysis with support score and best supporting snippets
        """
        if not evidence_snippets:
            return ClaimAnalysis(
                claim=claim,
                supported=False,
                support_score=0.0,
                supporting_snippets=[],
                missing_info="No evidence snippets provided"
            )
        
        best_score = 0.0
        best_snippets = []
        
        # Use normalized claim if available for better matching
        claim_text = claim.normalized or claim.text
        
        for snippet in evidence_snippets:
            entailment_prob = self._compute_entailment(claim_text, snippet.text)
            
            if entailment_prob > best_score:
                best_score = entailment_prob
            
            # Include snippet if it meets threshold
            if entailment_prob >= threshold:
                snippet.score = entailment_prob
                best_snippets.append(snippet)
        
        # Sort by score descending
        best_snippets.sort(key=lambda x: x.score, reverse=True)
        
        # Determine if claim is supported
        supported = best_score >= threshold
        
        missing_info = None
        if not supported:
            missing_info = f"Claim not supported by evidence (best score: {best_score:.2f})"
        
        return ClaimAnalysis(
            claim=claim,
            supported=supported,
            support_score=best_score,
            supporting_snippets=best_snippets,
            missing_info=missing_info
        )
    
    def _compute_entailment(self, premise: str, hypothesis: str) -> float:
        """Compute entailment probability between premise and hypothesis.
        
        In NLI for evidence coverage:
        - Premise: the evidence/passage text
        - Hypothesis: the claim to verify
        
        Args:
            premise: The evidence text
            hypothesis: The claim text
            
        Returns:
            Entailment probability (0.0 to 1.0)
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get entailment probability
            entailment_prob = probs[0][self.entailment_label_id].item()
        
        return entailment_prob
    
    def score_batch(
        self,
        claims: List[Claim],
        evidence_dict: dict,
        threshold: float = 0.7
    ) -> List[ClaimAnalysis]:
        """Score multiple claims in batch.
        
        Args:
            claims: List of claims to score
            evidence_dict: Dictionary mapping claims to their evidence snippets
            threshold: Minimum entailment probability
            
        Returns:
            List of ClaimAnalysis objects
        """
        results = []
        for claim in claims:
            evidence = evidence_dict.get(claim, [])
            analysis = self.score_claim(claim, evidence, threshold)
            results.append(analysis)
        return results
