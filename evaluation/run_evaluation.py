"""Evaluation runner script for Filewise Knowledge Assistant."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.config import settings
from app.models.schemas import AskRequest
from app.services.chunking import TextChunker
from app.services.document_loader import DocumentLoader
from app.services.embeddings import EmbeddingService
from app.services.llm_provider import create_llm_provider
from app.services.qa_service import QAService
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("eval")


def ingest_sample_documents(vector_store: VectorStore, embedding_service: EmbeddingService):
    """Ingest all sample documents into the evaluation vector store."""
    sample_dir = BASE_DIR / "data" / "sample_documents"
    chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)

    for doc_path in sample_dir.iterdir():
        if not doc_path.is_file():
            continue
        filename = doc_path.name
        if not DocumentLoader.is_supported(filename):
            continue

        if vector_store.is_document_indexed(filename):
            logger.info("Sample document '%s' already indexed.", filename)
            continue

        content = doc_path.read_bytes()
        pages = DocumentLoader.load_bytes(filename, content)
        chunks = chunker.chunk_document(filename, pages)
        embeddings = embedding_service.embed_texts([c.text for c in chunks])
        vector_store.add_chunks(chunks, embeddings)
        logger.info("Ingested '%s' (%d chunks).", filename, len(chunks))


def run_evaluation(output_file: Path = None):
    print("=" * 70)
    print("FILEWISE — COMPREHENSIVE EVALUATION RUNNER")
    print("=" * 70)

    # Isolated evaluation vector store
    eval_store_dir = BASE_DIR / "data" / "vector_store_eval"
    eval_store = VectorStore(storage_dir=eval_store_dir)
    eval_store.clear()

    embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL_NAME)

    print("\n[1/3] Ingesting sample documents into evaluation vector store...")
    ingest_sample_documents(eval_store, embedding_service)
    print(f"Total indexed chunks: {eval_store.total_chunks()}")

    retrieval_service = RetrievalService(eval_store, embedding_service)
    llm_provider = create_llm_provider(settings.LLM_PROVIDER, settings.LLM_API_KEY, settings.LLM_MODEL_NAME)
    qa_service = QAService(retrieval_service, llm_provider, default_top_k=4, default_threshold=0.35)

    questions_file = BASE_DIR / "evaluation" / "questions.json"
    with open(questions_file, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    print(f"\n[2/3] Executing {len(questions_data)} evaluation test questions...")

    total_questions = len(questions_data)
    correct_abstentions = 0
    total_unanswerable = 0
    correct_answerables = 0
    total_answerable = 0
    retrieval_source_hits = 0
    keyword_hits = 0

    results = []

    for q in questions_data:
        qid = q["id"]
        question_text = q["question"]
        expected_ans = q["expected_answerable"]
        expected_sources = q.get("expected_sources", [])
        expected_keywords = q.get("keywords", [])

        req = AskRequest(question=question_text)
        t0 = time.time()
        resp = qa_service.answer_question(req)
        latency = round((time.time() - t0) * 1000, 2)

        actual_sources = [s.document for s in resp.sources]

        # Evaluate Answerability / Abstention
        if not expected_ans:
            total_unanswerable += 1
            if not resp.answerable:
                correct_abstentions += 1
                status = "PASS (Correctly Abstained)"
            else:
                status = "FAIL (Hallucinated/Failed to Abstain)"
        else:
            total_answerable += 1
            if resp.answerable:
                correct_answerables += 1
                status = "PASS (Answered)"
            else:
                status = "FAIL (False Negative Abstention)"

        # Evaluate Source Document Retrieval Coverage
        source_covered = False
        if not expected_sources:
            source_covered = len(actual_sources) == 0
        else:
            # Check if expected sources overlap with actual sources
            matched = set(expected_sources).intersection(set(actual_sources))
            source_covered = len(matched) > 0
            if source_covered:
                retrieval_source_hits += 1

        # Evaluate Grounded Content / Keywords
        k_hit = False
        answer_lower = resp.answer.lower()
        if expected_keywords:
            k_hit = any(kw.lower() in answer_lower for kw in expected_keywords)
            if k_hit:
                keyword_hits += 1

        results.append({
            "id": qid,
            "type": q.get("type", "standard"),
            "question": question_text,
            "expected_answerable": expected_ans,
            "actual_answerable": resp.answerable,
            "status": status,
            "latency_ms": latency,
            "expected_sources": expected_sources,
            "actual_sources": actual_sources,
            "source_covered": source_covered,
            "keyword_hit": k_hit,
            "answer_preview": resp.answer[:140] + ("..." if len(resp.answer) > 140 else "")
        })

        print(f"[{qid}] {status} | Latency: {latency}ms | Sources: {actual_sources}")

    print("\n[3/3] Computing aggregate metrics...")

    abstention_accuracy = (correct_abstentions / total_unanswerable * 100) if total_unanswerable else 100.0
    answerable_accuracy = (correct_answerables / total_answerable * 100) if total_answerable else 100.0
    overall_accuracy = ((correct_abstentions + correct_answerables) / total_questions * 100)
    source_coverage_rate = (retrieval_source_hits / total_answerable * 100) if total_answerable else 100.0
    grounded_keyword_rate = (keyword_hits / total_questions * 100)

    report = {
        "summary": {
            "total_questions": total_questions,
            "total_answerable": total_answerable,
            "total_unanswerable": total_unanswerable,
            "overall_accuracy_percent": round(overall_accuracy, 2),
            "answerable_accuracy_percent": round(answerable_accuracy, 2),
            "abstention_accuracy_percent": round(abstention_accuracy, 2),
            "source_coverage_percent": round(source_coverage_rate, 2),
            "grounded_keyword_accuracy_percent": round(grounded_keyword_rate, 2),
        },
        "results": results
    }

    print("\n" + "=" * 70)
    print("EVALUATION REPORT SUMMARY:")
    print(f"  • Total Questions Evaluated:  {total_questions}")
    print(f"  • Overall Decision Accuracy:  {overall_accuracy:.1f}%")
    print(f"  • Correct Abstention Rate:    {abstention_accuracy:.1f}% ({correct_abstentions}/{total_unanswerable})")
    print(f"  • Answerable Recall:          {answerable_accuracy:.1f}% ({correct_answerables}/{total_answerable})")
    print(f"  • Source Retrieval Coverage:  {source_coverage_rate:.1f}% ({retrieval_source_hits}/{total_answerable})")
    print("=" * 70)

    if output_file is None:
        output_file = BASE_DIR / "evaluation" / "evaluation_report.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved detailed evaluation report to {output_file}")

    # Clean up eval store
    eval_store.clear()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Filewise evaluation suite.")
    parser.add_argument("--output", type=str, default=None, help="Output path for evaluation report")
    args = parser.parse_args()
    out = Path(args.output) if args.output else None
    run_evaluation(out)
