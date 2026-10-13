"""Evidence-grounded Question Answering service."""

import logging
import re
from typing import List, Optional, Tuple
from app.models.schemas import AskRequest, AskResponse, DocumentChunk, SourceCitation
from app.services.llm_provider import LLMProvider
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

UNAVAILABLE_MESSAGE = "I could not find sufficient information in the uploaded documents to answer this question."


class QAService:
    """Coordinates retrieval, relevance thresholding, LLM generation, and source excerpt attribution."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_provider: LLMProvider,
        default_top_k: int = 4,
        default_threshold: float = 0.35,
    ):
        self.retrieval_service = retrieval_service
        self.llm_provider = llm_provider
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold

    def _extract_best_excerpt(self, chunk_text: str, question: str, max_words: int = 45) -> str:
        """Extract a representative, verbatim excerpt from the chunk matching the question."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", chunk_text) if s.strip()]
        if not sentences:
            return chunk_text[:200]

        q_words = set(re.findall(r"\w+", question.lower()))
        best_sentence = sentences[0]
        max_overlap = -1

        for s in sentences:
            s_words = set(re.findall(r"\w+", s.lower()))
            overlap = len(q_words.intersection(s_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = s

        # Return the best sentence or trimmed version
        words = best_sentence.split()
        if len(words) <= max_words:
            return best_sentence
        return " ".join(words[:max_words]) + "..."

    def answer_question(self, request: AskRequest) -> AskResponse:
        """Process a question, retrieve relevant chunks, generate grounded answer, and format sources."""
        top_k = request.top_k or self.default_top_k
        threshold = request.relevance_threshold or self.default_threshold

        retrieved_results: List[Tuple[DocumentChunk, float]] = self.retrieval_service.retrieve(
            query=request.question,
            top_k=top_k,
            relevance_threshold=threshold,
        )

        # Early return if no relevant chunks meet threshold
        if not retrieved_results:
            return AskResponse(
                question=request.question,
                answer=UNAVAILABLE_MESSAGE,
                answerable=False,
                sources=[],
                message="No document excerpts met the relevance threshold.",
            )

        context_chunks = [chunk for chunk, _ in retrieved_results]

        # Call LLM
        answer = self.llm_provider.generate_grounded_answer(
            question=request.question,
            context_chunks=context_chunks,
        )

        # Check if LLM determined the question is unanswerable from the context
        is_unanswerable = (
            UNAVAILABLE_MESSAGE.lower() in answer.lower()
            or "sufficient information in the uploaded documents" in answer.lower()
            or "information is unavailable in the uploaded documents" in answer.lower()
            or not answer.strip()
        )

        if is_unanswerable:
            return AskResponse(
                question=request.question,
                answer=UNAVAILABLE_MESSAGE,
                answerable=False,
                sources=[],
                message="The retrieved documents do not contain sufficient evidence to answer this question.",
            )

        # Construct genuine source citations
        sources: List[SourceCitation] = []
        for chunk, score in retrieved_results:
            excerpt = self._extract_best_excerpt(chunk.text, request.question)
            sources.append(
                SourceCitation(
                    document=chunk.document_name,
                    page=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    excerpt=excerpt,
                    relevance_score=round(score, 4),
                )
            )

        return AskResponse(
            question=request.question,
            answer=answer,
            answerable=True,
            sources=sources,
            message=None,
        )
