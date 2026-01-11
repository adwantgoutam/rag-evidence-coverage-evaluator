"""ECE Optional Module.

Author: Goutam Adwant (gadwant)

Optional components for the ECE framework, including LLM-based
evaluation using external API providers (OpenAI, Anthropic).
"""

from ece.optional.llm_judge import LLMJudge

__all__ = ["LLMJudge"]
