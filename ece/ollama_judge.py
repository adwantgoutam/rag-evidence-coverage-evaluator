"""Module for Ollama-based LLM claim evaluation (Mode B).

Author: Goutam Adwant (gadwant)

This module implements Mode B evaluation using local Ollama LLM models
for sophisticated claim-evidence reasoning. It provides an alternative
to NLI-based scoring with more nuanced semantic understanding.
"""

import json
import requests
from typing import List, Optional, Dict, Any
from ece.models import Claim, SupportingSnippet, ClaimAnalysis


class OllamaJudge:
    """Uses local Ollama LLM to judge claim support (Mode B).
    
    Leverages locally-hosted LLM models through Ollama for evaluating
    claim-evidence relationships with sophisticated reasoning capabilities.
    
    Attributes:
        model: Ollama model name (e.g., "mistral:latest", "llama3:latest")
        base_url: Ollama API base URL
        temperature: Sampling temperature (0.0 for deterministic output)
    """
    
    def __init__(
        self,
        model: str = "mistral:latest",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0
    ):
        """Initialize the Ollama judge.
        
        Args:
            model: Ollama model name (e.g., "mistral:latest", "llama3:latest")
            base_url: Ollama API base URL (default: http://localhost:11434)
            temperature: Sampling temperature (0.0 for deterministic output)
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        
    def score_claim(
        self,
        claim: Claim,
        evidence_snippets: List[SupportingSnippet],
        threshold: float = 0.7
    ) -> ClaimAnalysis:
        """Judge a claim using Ollama LLM.
        
        Sends the claim and evidence to the Ollama model for evaluation,
        receiving a structured judgment on whether the claim is supported.
        
        Args:
            claim: The claim to evaluate
            evidence_snippets: List of evidence snippets to check against
            threshold: Minimum score to consider supported (for consistency)
            
        Returns:
            ClaimAnalysis with LLM judgment results
        """
        if not evidence_snippets:
            return ClaimAnalysis(
                claim=claim,
                supported=False,
                support_score=0.0,
                supporting_snippets=[],
                missing_info="No evidence snippets provided"
            )
        
        # Format evidence for the prompt
        evidence_text = self._format_evidence(evidence_snippets)
        
        # Create structured prompt
        prompt = self._create_prompt(claim.text, evidence_text)
        
        # Call Ollama API
        response = self._call_ollama(prompt)
        
        # Parse JSON response
        judgment = self._parse_response(response)
        
        # Extract judgment fields
        supported = judgment.get("supported", False)
        support_score = judgment.get("confidence", 1.0 if supported else 0.0)
        supporting_texts = judgment.get("supporting_texts", [])
        missing_info = judgment.get("missing_info")
        
        # Match supporting texts to evidence snippets
        supporting_snippets = []
        for text in supporting_texts:
            for snippet in evidence_snippets:
                if text.lower() in snippet.text.lower() or snippet.text.lower() in text.lower():
                    snippet.score = support_score if supported else 0.0
                    supporting_snippets.append(snippet)
                    break
        
        # Use top snippet if claim supported but no text matches found
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
        """Format evidence snippets for the prompt.
        
        Args:
            snippets: List of evidence snippets
            
        Returns:
            Formatted string with labeled passages
        """
        formatted = []
        for snippet in snippets:
            formatted.append(f"[Passage {snippet.passage_id}]: {snippet.text}")
        return "\n\n".join(formatted)
    
    def _create_prompt(self, claim: str, evidence: str) -> str:
        """Create the LLM evaluation prompt.
        
        Args:
            claim: The claim text to evaluate
            evidence: Formatted evidence text
            
        Returns:
            Complete prompt string for the LLM
        """
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
    
    def _call_ollama(self, prompt: str) -> str:
        """Call the Ollama API.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            Raw response text from the model
            
        Raises:
            RuntimeError: If the API call fails
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            raise RuntimeError(f"Failed to call Ollama API: {e}")
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response.
        
        Handles various response formats including markdown code blocks.
        
        Args:
            response: Raw response text from the LLM
            
        Returns:
            Parsed judgment dictionary
        """
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
            # Fallback: extract basic information from text
            return {
                "supported": "true" in response.lower() or "supported" in response.lower(),
                "confidence": 0.5,
                "supporting_texts": [],
                "missing_info": "Failed to parse LLM response"
            }
