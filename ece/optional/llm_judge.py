"""Module for LLM-based claim evaluation (Mode B - Optional).

Author: Goutam Adwant (gadwant)

This module provides LLM-based evaluation using external API providers
(OpenAI, Anthropic). It requires additional dependencies and API keys.

To use this module, install the required dependencies:
    pip install openai>=1.0.0  # or anthropic>=0.7.0
"""

import json
from typing import List, Optional, Dict, Any
from ece.models import Claim, SupportingSnippet, ClaimAnalysis


class LLMJudge:
    """Uses external LLM APIs to judge claim support (Mode B).
    
    This class provides an alternative to NLI-based scoring using
    commercial LLM APIs for more sophisticated reasoning.
    
    Attributes:
        provider: LLM provider ("openai" or "anthropic")
        model: Model name (e.g., "gpt-4", "claude-3-opus")
        temperature: Sampling temperature (0.0 for deterministic)
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.0
    ):
        """Initialize the LLM judge.
        
        Args:
            provider: LLM provider ("openai" or "anthropic")
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo", "claude-3-opus")
            api_key: API key (or set environment variable)
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        
        if provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'")
    
    def score_claim(
        self,
        claim: Claim,
        evidence_snippets: List[SupportingSnippet],
        threshold: float = 0.7
    ) -> ClaimAnalysis:
        """Judge a claim using LLM.
        
        Args:
            claim: The claim to evaluate
            evidence_snippets: List of evidence snippets
            threshold: Minimum score to consider supported (for consistency with NLI)
            
        Returns:
            ClaimAnalysis with LLM judgment
        """
        if not evidence_snippets:
            return ClaimAnalysis(
                claim=claim,
                supported=False,
                support_score=0.0,
                supporting_snippets=[],
                missing_info="No evidence snippets provided"
            )
        
        # Format evidence
        evidence_text = self._format_evidence(evidence_snippets)
        
        # Create prompt
        prompt = self._create_prompt(claim.text, evidence_text)
        
        # Call LLM
        response = self._call_llm(prompt)
        
        # Parse response
        judgment = self._parse_response(response)
        
        # Build result
        supported = judgment.get("supported", False)
        support_score = judgment.get("confidence", 1.0 if supported else 0.0)
        supporting_texts = judgment.get("supporting_texts", [])
        missing_info = judgment.get("missing_info")
        
        # Match supporting texts to snippets
        supporting_snippets = []
        for text in supporting_texts:
            for snippet in evidence_snippets:
                if text.lower() in snippet.text.lower() or snippet.text.lower() in text.lower():
                    snippet.score = support_score if supported else 0.0
                    supporting_snippets.append(snippet)
                    break
        
        # If no matches found but claim is supported, use top evidence snippet
        if supported and not supporting_snippets and evidence_snippets:
            top_snippet = evidence_snippets[0]
            top_snippet.score = support_score
            supporting_snippets = [top_snippet]
        
        return ClaimAnalysis(
            claim=claim,
            supported=supported,
            support_score=support_score,
            supporting_snippets=supporting_snippets,
            missing_info=missing_info
        )
    
    def _format_evidence(self, snippets: List[SupportingSnippet]) -> str:
        """Format evidence snippets for the prompt."""
        formatted = []
        for snippet in snippets:
            formatted.append(f"[Passage {snippet.passage_id}]: {snippet.text}")
        return "\n\n".join(formatted)
    
    def _create_prompt(self, claim: str, evidence: str) -> str:
        """Create the LLM prompt."""
        return f"""You are evaluating whether a claim is supported by evidence.

CLAIM: {claim}

EVIDENCE:
{evidence}

Evaluate whether the claim is FULLY supported by the provided evidence. Consider:
1. Is the claim directly stated or clearly implied by the evidence?
2. Are all parts of the claim supported?
3. If not fully supported, what information is missing?

Respond in JSON format:
{{
    "supported": true/false,
    "confidence": 0.0-1.0,
    "supporting_texts": ["exact quotes from evidence that support the claim"],
    "missing_info": "description of missing information (if not supported)"
}}

Respond ONLY with valid JSON, no other text."""

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        if self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that evaluates claim-evidence pairs. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"}
                )
            except Exception:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that evaluates claim-evidence pairs. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature
                )
            return response.choices[0].message.content
        else:  # anthropic
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response."""
        try:
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            judgment = json.loads(response)
            return judgment
        except json.JSONDecodeError:
            return {
                "supported": "true" in response.lower() or "supported" in response.lower(),
                "confidence": 0.5,
                "supporting_texts": [],
                "missing_info": "Failed to parse LLM response"
            }
