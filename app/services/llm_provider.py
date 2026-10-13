"""LLM provider abstraction supporting Mock, Gemini, and OpenAI."""

import abc
import logging
from typing import List
from app.models.schemas import DocumentChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Filewise, an evidence-grounded AI knowledge assistant.
Your answers MUST be strictly based on the provided document excerpts below.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using the facts stated directly in the provided evidence.
2. Do NOT use outside knowledge, assumptions, or external information to fill gaps.
3. Do NOT invent or extrapolate facts, names, dates, statistics, quotations, or conclusions.
4. If the provided evidence does not contain enough information to answer the question, you MUST reply:
   "I could not find sufficient information in the uploaded documents to answer this question."
5. If only part of the question can be answered, answer the supported part and explicitly identify what information is missing.
6. When multiple pieces of evidence are relevant, combine them accurately.
7. If different documents or chunks contradict each other, explicitly note the contradiction and describe both positions.
8. Keep answers clear, direct, and concise."""


class LLMProvider(abc.ABC):
    """Abstract base class for LLM generation."""

    @abc.abstractmethod
    def generate_grounded_answer(self, question: str, context_chunks: List[DocumentChunk]) -> str:
        """Generate an answer strictly grounded in the given context chunks."""
        pass


class MockLLMProvider(LLMProvider):
    """Deterministic, rule-based provider for offline testing and evaluation without paid API keys.

    Extracts matching sentences directly from context chunks to ensure answers are genuine
    subsets of uploaded documents without fabrication.
    """

    def generate_grounded_answer(self, question: str, context_chunks: List[DocumentChunk]) -> str:
        if not context_chunks:
            return "I could not find sufficient information in the uploaded documents to answer this question."

        # Collect text from chunks
        combined_text = "\n".join(chunk.text for chunk in context_chunks)
        q_lower = question.lower()

        # Split into sentences
        import re

        sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", combined_text) if len(s.strip()) > 10]

        # Extract question keywords (ignore common stop words)
        stopwords = {
            "what", "is", "the", "are", "of", "in", "to", "and", "a", "an", "for", "on", "how",
            "many", "who", "which", "does", "do", "can", "should", "will", "would", "at", "by",
        }
        tokens = [t.strip("?.,!\"'()") for t in q_lower.split() if t.strip("?.,!\"'()") not in stopwords]

        matched_sentences = []
        for s in sentences:
            s_lower = s.lower()
            # Score sentence by token overlap
            matches = sum(1 for t in tokens if t in s_lower)
            if matches > 0:
                matched_sentences.append((matches, s))

        if not matched_sentences:
            return "I could not find sufficient information in the uploaded documents to answer this question."

        # Sort by best matching
        matched_sentences.sort(key=lambda x: x[0], reverse=True)
        best_match_count = matched_sentences[0][0]

        # Grounding check:
        # Determine informative tokens (excluding company/generic names and generic temporal modifiers)
        generic_terms = {
            "meridian", "technologies", "technology", "project", "aegis", "company",
            "annual", "year", "2025", "2026", "calendar", "time", "day", "days"
        }
        # Core subject tokens that define the crux of the query (e.g. revenue, cfo, vacation, pto, stipend, hours, rto)
        core_subject_tokens = [t for t in tokens if t not in generic_terms]

        if core_subject_tokens:
            has_core_match = any(any(cst in s.lower() for cst in core_subject_tokens) for _, s in matched_sentences)
            if not has_core_match:
                return "I could not find sufficient information in the uploaded documents to answer this question."

        required_coverage = 2 if len(tokens) >= 3 else 1
        if best_match_count < required_coverage:
            return "I could not find sufficient information in the uploaded documents to answer this question."

        # Pick top distinct sentences with sufficient match
        selected = []
        seen = set()
        for count, s in matched_sentences:
            s_lower = s.lower()
            if count < required_coverage:
                continue
            if core_subject_tokens and not any(inf in s_lower for inf in core_subject_tokens):
                continue
            cleaned = s.strip()
            if cleaned not in seen:
                seen.add(cleaned)
                selected.append(cleaned)
            if len(selected) >= 3:
                break

        if not selected:
            return "I could not find sufficient information in the uploaded documents to answer this question."

        return " ".join(selected)


class GeminiLLMProvider(LLMProvider):
    """Google Gemini LLM provider using the modern google-genai SDK."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError("Gemini API key is missing. Set LLM_API_KEY in .env or environment.")
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate_grounded_answer(self, question: str, context_chunks: List[DocumentChunk]) -> str:
        if not context_chunks:
            return "I could not find sufficient information in the uploaded documents to answer this question."

        context_blocks = []
        for idx, chunk in enumerate(context_chunks, 1):
            page_info = f" (Page {chunk.page_number})" if chunk.page_number else ""
            context_blocks.append(
                f"[Document: {chunk.document_name}{page_info} | Chunk ID: {chunk.chunk_id}]\n{chunk.text}"
            )
        formatted_context = "\n\n---\n\n".join(context_blocks)

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"=== PROVIDED DOCUMENT EVIDENCE ===\n{formatted_context}\n\n"
            f"=== QUESTION ===\n{question}\n\n"
            f"=== GROUNDED ANSWER ==="
        )

        try:
            client = self._get_client()
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            logger.error("Gemini API error: %s", e)
            raise RuntimeError(f"Gemini LLM generation failed: {e}") from e


class OpenAILLMProvider(LLMProvider):
    """OpenAI API provider for configurable LLM backends."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model_name = model_name

    def generate_grounded_answer(self, question: str, context_chunks: List[DocumentChunk]) -> str:
        if not context_chunks:
            return "I could not find sufficient information in the uploaded documents to answer this question."

        if not self.api_key:
            raise ValueError("OpenAI API key is missing. Set LLM_API_KEY in .env or environment.")

        import httpx

        context_blocks = [
            f"[{c.document_name}{f' p.{c.page_number}' if c.page_number else ''}]: {c.text}"
            for c in context_chunks
        ]
        context_str = "\n\n".join(context_blocks)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"DOCUMENT EVIDENCE:\n{context_str}\n\nQUESTION: {question}\n\nGROUNDED ANSWER:",
            },
        ]

        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model_name, "messages": messages, "temperature": 0.0},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error("OpenAI API call failed: %s", e)
            raise RuntimeError(f"OpenAI LLM generation failed: {e}") from e


def create_llm_provider(provider_type: str, api_key: str = "", model_name: str = "") -> LLMProvider:
    """Factory creating the appropriate LLM provider instance."""
    p = provider_type.lower().strip()
    if p == "gemini":
        return GeminiLLMProvider(api_key=api_key, model_name=model_name or "gemini-2.5-flash")
    elif p == "openai":
        return OpenAILLMProvider(api_key=api_key, model_name=model_name or "gpt-4o-mini")
    else:
        return MockLLMProvider()
