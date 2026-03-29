"""
LLM Provider abstraction — supports Ollama, Groq, OpenAI backends and MCP via LangChain.
Integrated with LangSmith / LangGraph.

Selected via the ``LLM_PROVIDER`` environment variable.
Provides ``summarize()`` and ``classify()`` methods.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from app.config import get_settings

logger = logging.getLogger(__name__)

class LLMProvider:
    """Abstract base for LLM providers using LangChain."""
    def __init__(self, llm):
        self.llm = llm

    async def _generate(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        resp = await self.llm.ainvoke(messages)
        return resp.content

    async def summarize(self, text: str, mode: str = "chunk") -> str:
        prompt = self._build_summarize_prompt(text, mode)
        return await self._generate(prompt)

    async def classify(self, text: str, labels: list[str]) -> dict[str, Any]:
        prompt = self._build_classify_prompt(text, labels)
        response = await self._generate(prompt)
        return self._parse_classify_response(response, text, labels)

    def _build_summarize_prompt(self, text: str, mode: str) -> str:
        if mode == "final":
            return (
                "You are given multiple summaries of document chunks. "
                "Produce a single coherent summary that captures all key points.\n\n"
                f"Chunk summaries:\n{text}\n\n"
                "Final summary:"
            )
        return (
            "Summarize the following text concisely, capturing the main points:\n\n"
            f"{text}\n\nSummary:"
        )

    def _build_classify_prompt(self, text: str, labels: list[str]) -> str:
        labels_str = ", ".join(f'"{l}"' for l in labels)
        return (
            f"Classify the following text into exactly one of these labels: [{labels_str}].\n\n"
            f"Text: {text}\n\n"
            "Respond with ONLY a JSON object: {\"label\": \"chosen_label\", \"confidence\": 0.0-1.0}"
        )

    def _parse_classify_response(self, response: str, text: str, labels: list[str]) -> dict[str, Any]:
        """Parse the LLM classification response into structured output."""
        try:
            result = json.loads(response.strip())
            return {
                "text": text[:100],
                "label": result.get("label", labels[0]),
                "confidence": float(result.get("confidence", 0.5)),
            }
        except (json.JSONDecodeError, ValueError):
            response_lower = response.lower()
            for label in labels:
                if label.lower() in response_lower:
                    return {"text": text[:100], "label": label, "confidence": 0.5}
            return {"text": text[:100], "label": labels[0], "confidence": 0.1}

_provider_instance = None

def get_llm_provider() -> LLMProvider:
    """
    Return the correct provider via LangChain.

    Resolution order:
      1. If GROQ_API_KEY is present -> Groq.
      2. If LLM_PROVIDER is explicitly set -> honour it.
      3. Fallback -> Ollama (local).
    """
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    settings = get_settings()
    provider_name = settings.LLM_PROVIDER.lower()

    if getattr(settings, "XAI_API_KEY", "") and provider_name not in ("ollama", "claude"):
        logger.info("LLM provider: Grok (XAI_API_KEY detected)")
        llm = ChatOpenAI(api_key=settings.XAI_API_KEY, base_url="https://api.x.ai/v1", model="grok-2-latest", temperature=0.0)
        _provider_instance = LLMProvider(llm)
        return _provider_instance

    if settings.GROQ_API_KEY and provider_name != "ollama":
        logger.info("LLM provider: Groq (GROQ_API_KEY detected)")
        llm = ChatGroq(api_key=settings.GROQ_API_KEY, model_name=settings.GROQ_MODEL, temperature=0.3)
        _provider_instance = LLMProvider(llm)
        return _provider_instance

    if provider_name == "mcp":
        logger.info("Initializing LLM provider: MCP (Model Context Protocol)")
        llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model_name=settings.OPENAI_MODEL, temperature=0.3)
        _provider_instance = LLMProvider(llm)
    elif provider_name == "ollama":
        logger.info("Initializing LLM provider: ollama")
        llm = ChatOllama(base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_MODEL, temperature=0.3)
        _provider_instance = LLMProvider(llm)
    elif provider_name == "openai":
        logger.info("Initializing LLM provider: openai")
        llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY, model_name=settings.OPENAI_MODEL, temperature=0.3)
        _provider_instance = LLMProvider(llm)
    elif provider_name == "azure_openai":
        logger.info("Initializing LLM provider: azure_openai")
        llm = AzureChatOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.3
        )
        _provider_instance = LLMProvider(llm)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider_name}")

    return _provider_instance
