"""
CLI Script: Ingest technical documents into the persistent RAG Vector Database.

Usage:
    python scripts/ingest_documents.py --path data/rag/documents/ --config configs/rag.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.config import RAGConfig
from src.rag.ingestion.pipeline import DocumentIngestionPipeline
from src.rag.chunking.chunker import TechnicalDocumentChunker
from src.rag.embeddings.model import create_embedding_model
from src.rag.vectorstore.store import NumpyFlatVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ingest_documents")


def main():
    parser = argparse.ArgumentParser(description="Ingest technical documents into RAG vector database.")
    parser.add_argument(
        "--path",
        type=str,
        default="data/rag/documents",
        help="Directory containing PDF/TXT/MD technical manuals."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rag.yaml",
        help="Path to RAG YAML configuration file."
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force reindexing of previously indexed documents ignoring hash manifest."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Optional override for vector store collection name."
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        logger.info(f"Loading configuration from '{config_path}'")
        config = RAGConfig.from_yaml(config_path)
    else:
        logger.warning(f"Config '{config_path}' not found. Using default RAGConfig.")
        config = RAGConfig()

    if args.collection:
        config.vector_store.collection_name = args.collection

    doc_dir = Path(args.path)
    if not doc_dir.exists():
        logger.error(f"Document directory '{doc_dir}' does not exist.")
        sys.exit(1)

    logger.info("Initializing Ingestion Pipeline, Chunker, Embedding Model, and Vector Store...")
    pipeline = DocumentIngestionPipeline(config.ingestion, manifest_path=config.manifest_path)
    chunker = TechnicalDocumentChunker(config.chunking)
    embedding_model = create_embedding_model(config.embedding)
    vector_store = NumpyFlatVectorStore(config.vector_store)

    logger.info(f"Scanning document directory '{doc_dir}'...")
    ingest_result = pipeline.ingest_directory(doc_dir, force_reindex=args.force_reindex)

    parsed_docs = ingest_result["parsed_documents"]
    total_new_chunks = 0

    print("\n=======================================================")
    print("           DOCUMENT INGESTION REPORT                   ")
    print("=======================================================")
    print(f"Documents Discovered:   {ingest_result['documents_found']}")
    print(f"Documents Parsed:       {ingest_result['documents_parsed']}")
    print(f"Documents Skipped:      {ingest_result['documents_skipped']} (unmodified hashes)")
    print(f"Total Pages Extracted:  {ingest_result['total_pages_extracted']}")
    print(f"Failed Documents:       {len(ingest_result['failed_files'])}")

    if ingest_result["failed_files"]:
        print("\nFailures:")
        for failed_path, err in ingest_result["failed_files"]:
            print(f"  - {failed_path}: {err}")

    # Process chunks and embeddings for newly parsed documents
    for meta, pages in parsed_docs:
        chunks = chunker.chunk_document(meta, pages)
        if not chunks:
            logger.warning(f"No text chunks generated for document '{meta.document_name}'")
            continue

        logger.info(f"Generating embeddings for {len(chunks)} chunks from '{meta.document_name}'...")
        chunk_texts = [c.text for c in chunks]
        embeddings = embedding_model.embed_documents(chunk_texts)

        vector_store.add_chunks(chunks, embeddings)
        total_new_chunks += len(chunks)

    if total_new_chunks > 0:
        vector_store.save()
        logger.info(f"Persisted vector database with {vector_store.count()} total indexed chunks.")

    print(f"New Chunks Indexed:     {total_new_chunks}")
    print(f"Total Database Chunks:  {vector_store.count()}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
