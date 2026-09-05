"""
Unit and Integration Tests for Phase 6 Technical Knowledge RAG Subsystem.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.rag.chunking.chunker import TechnicalDocumentChunker
from src.rag.config import (
    ChunkingConfig,
    DocumentIngestionConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from src.rag.embeddings.model import (
    DeterministicDenseEmbeddingModel,
)
from src.rag.evaluation.evaluator import EvaluationSample, RAGEvaluator
from src.rag.ingestion.parser import DocumentParser, compute_file_hash
from src.rag.ingestion.pipeline import DocumentIngestionPipeline
from src.rag.retrieval.retriever import TechnicalRetriever
from src.rag.schema import (
    DocumentChunk,
    DocumentMetadata,
    RawDocumentPage,
)
from src.rag.vectorstore.store import NumpyFlatVectorStore


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_schema_creation_and_serialization():
    chunk = DocumentChunk.create(
        document_id="doc_123",
        document_name="pump_manual.pdf",
        source_path="/path/pump_manual.pdf",
        page_number=3,
        text="Vibration limit is 4.5 mm/s RMS.",
        chunk_index=0,
        section="Section 3: Limits",
        equipment_type="pump",
    )
    assert chunk.document_id == "doc_123"
    assert chunk.page_number == 3
    assert chunk.equipment_type == "pump"
    d = chunk.to_dict()
    assert d["chunk_id"].startswith("doc_123_p3_c0_")

    restored = DocumentChunk.from_dict(d)
    assert restored.chunk_id == chunk.chunk_id
    assert restored.text == chunk.text


def test_document_parser_text_and_markdown(temp_dir):
    md_file = temp_dir / "test_manual.md"
    md_file.write_text(
        "# MOTOR MANUAL\n\n--- PAGE 1 ---\n1.0 General Specifications\nOperating temp is 60C.\n\n--- PAGE 2 ---\n2.0 Lubrication\nApply grease every 500 hours.",
        encoding="utf-8",
    )

    parser = DocumentParser()
    meta, pages = parser.parse_document(md_file, extra_metadata={"equipment_type": "motor"})

    assert meta.document_name == "test_manual.md"
    assert meta.file_type == "md"
    assert len(pages) >= 2
    assert meta.equipment_type == "motor"
    assert compute_file_hash(md_file) == meta.file_hash


def test_scanned_pdf_sparse_detection(temp_dir):
    sparse_file = temp_dir / "sparse.txt"
    sparse_file.write_text("Hi", encoding="utf-8")

    parser = DocumentParser(DocumentIngestionConfig(min_page_chars_threshold=30))
    meta, pages = parser.parse_document(sparse_file)
    assert len(pages) == 1


def test_technical_chunking_with_headings_and_tables(temp_dir):
    chunker = TechnicalDocumentChunker(ChunkingConfig(chunk_size=200, chunk_overlap=30))
    meta = DocumentMetadata(
        document_id="doc_01",
        document_name="spec.pdf",
        source_path="/test/spec.pdf",
        file_type="pdf",
        file_hash="abc123",
        file_size_bytes=1024,
        num_pages=1,
        equipment_type="compressor",
    )
    pages = [
        RawDocumentPage(
            page_number=1,
            text=(
                "SECTION 1: COMPRESSOR LIMITS\n\n"
                "Parameter | Normal | Alarm | Trip\n"
                "Pressure  | 100 psi| 120 psi| 150 psi\n\n"
                "Paragraph describing regular maintenance procedures for air filtration and oil separators. "
                "Ensure intake valves are cleaned weekly."
            ),
            section_title="SECTION 1: COMPRESSOR LIMITS",
        )
    ]
    chunks = chunker.chunk_document(meta, pages)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.document_name == "spec.pdf"
        assert c.page_number == 1
        assert c.equipment_type == "compressor"


def test_deterministic_embedding_model():
    model = DeterministicDenseEmbeddingModel(embedding_dim=256, normalize=True)
    assert model.embedding_dim == 256

    t1 = "Bearing high frequency defect at BPFI."
    t2 = "Bearing high frequency defect at BPFI."
    t3 = "Centrifugal pump cavitation and suction head."

    e1 = model.embed_text(t1)
    e2 = model.embed_text(t2)
    e3 = model.embed_text(t3)

    assert e1.shape == (256,)
    assert np.allclose(e1, e2)
    assert np.isclose(np.linalg.norm(e1), 1.0)
    assert not np.allclose(e1, e3)


def test_vector_store_persistence_and_filtering(temp_dir):
    cfg = VectorStoreConfig(
        persist_directory=str(temp_dir / "store"), collection_name="test_collection", distance_metric="cosine"
    )
    store = NumpyFlatVectorStore(cfg)
    emb_model = DeterministicDenseEmbeddingModel(embedding_dim=128)

    c1 = DocumentChunk.create(
        document_id="doc_motor",
        document_name="motor.pdf",
        source_path="motor.pdf",
        page_number=1,
        text="Motor rotor unbalance 1X vibration limit is 4.5 mm/s.",
        chunk_index=0,
        equipment_type="motor",
    )
    c2 = DocumentChunk.create(
        document_id="doc_pump",
        document_name="pump.pdf",
        source_path="pump.pdf",
        page_number=2,
        text="Centrifugal pump cavitation produces broadband acoustic hiss.",
        chunk_index=0,
        equipment_type="pump",
    )

    embs = emb_model.embed_documents([c1.text, c2.text])
    store.add_chunks([c1, c2], embs)
    store.save()

    new_store = NumpyFlatVectorStore(cfg)
    assert new_store.count() == 2

    q_emb = emb_model.embed_text("vibration")
    res_motor = new_store.search(q_emb, top_k=5, filters={"equipment_type": "motor"})
    assert len(res_motor) == 1
    assert res_motor[0][0].equipment_type == "motor"

    res_pump = new_store.search(q_emb, top_k=5, filters={"equipment_type": "pump"})
    assert len(res_pump) == 1
    assert res_pump[0][0].equipment_type == "pump"


def test_retriever_similarity_threshold_and_context_construction(temp_dir):
    cfg = VectorStoreConfig(persist_directory=str(temp_dir / "retrieval_store"))
    store = NumpyFlatVectorStore(cfg)
    emb_model = DeterministicDenseEmbeddingModel(embedding_dim=128)

    c1 = DocumentChunk.create(
        document_id="doc_sops",
        document_name="maintenance_sop.txt",
        source_path="maintenance_sop.txt",
        page_number=5,
        text="Inspect bearing grease discoloration and measure temperature every shift.",
        chunk_index=0,
        section="5.0 Bearing Lubrication",
        equipment_type="bearing",
    )
    store.add_chunks([c1], emb_model.embed_documents([c1.text]))

    ret_cfg = RetrievalConfig(top_k=5, similarity_threshold=0.10, enable_hybrid=True)
    retriever = TechnicalRetriever(store, emb_model, ret_cfg)

    # Completely unrelated query should fail similarity threshold if threshold is high
    unrelated_res = retriever.retrieve("recipe for baking chocolate cake", similarity_threshold=0.95)
    assert len(unrelated_res) == 0

    # Relevant query
    relevant_res = retriever.retrieve("bearing grease discoloration and temperature")
    assert len(relevant_res) == 1
    assert relevant_res[0].page_number == 5

    # Structured context builder
    context = retriever.build_evidence_context("bearing lubrication inspection", top_k=3)
    structured_text = context.to_structured_text()
    assert "TECHNICAL KNOWLEDGE RETRIEVAL EVIDENCE" in structured_text
    assert "PAGE: 5" in structured_text


def test_incremental_ingestion_pipeline_skips_unchanged_files(temp_dir):
    docs_dir = temp_dir / "docs"
    docs_dir.mkdir()
    f1 = docs_dir / "manual_v1.txt"
    f1.write_text("Initial manual revision text content.", encoding="utf-8")

    manifest = temp_dir / "manifest.json"
    pipeline = DocumentIngestionPipeline(manifest_path=manifest)

    # First ingestion
    res1 = pipeline.ingest_directory(docs_dir)
    assert res1["documents_parsed"] == 1
    assert res1["documents_skipped"] == 0

    # Second ingestion without modification
    res2 = pipeline.ingest_directory(docs_dir)
    assert res2["documents_parsed"] == 0
    assert res2["documents_skipped"] == 1

    # Modify file content
    f1.write_text("Modified revision with updated vibration tolerance.", encoding="utf-8")
    res3 = pipeline.ingest_directory(docs_dir)
    assert res3["documents_parsed"] == 1
    assert res3["documents_skipped"] == 0


def test_rag_evaluation_framework():
    store = NumpyFlatVectorStore(VectorStoreConfig(persist_directory="tmp/test_eval_store"))
    emb_model = DeterministicDenseEmbeddingModel(embedding_dim=64)

    chunk = DocumentChunk.create(
        document_id="eval_doc_01",
        document_name="fan_manual.pdf",
        source_path="fan_manual.pdf",
        page_number=4,
        text="Fan blade loose mountings cause high axial vibration.",
        chunk_index=0,
        section="BLADE INSPECTION",
    )
    store.add_chunks([chunk], emb_model.embed_documents([chunk.text]))

    retriever = TechnicalRetriever(store, emb_model, RetrievalConfig(similarity_threshold=0.01))
    evaluator = RAGEvaluator(retriever)

    samples = [
        EvaluationSample(
            query_id="TEST_Q1",
            query="What causes fan blade axial vibration?",
            target_document_name="fan_manual.pdf",
            target_page=4,
            target_section="BLADE INSPECTION",
        )
    ]

    metrics = evaluator.evaluate_benchmark(samples, top_k=3)
    assert metrics.total_queries == 1
    assert metrics.hit_rate_at_1 == 1.0
    assert metrics.mrr == 1.0
