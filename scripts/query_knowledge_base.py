"""
CLI Script: Query the Technical Knowledge RAG Base.

Usage:
    python scripts/query_knowledge_base.py --query "What are the recommended bearing inspection steps?"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.config import RAGConfig
from src.rag.retrieval.retriever import TechnicalRetriever
from src.rag.embeddings.model import create_embedding_model
from src.rag.vectorstore.store import NumpyFlatVectorStore


def main():
    parser = argparse.ArgumentParser(description="Query technical maintenance manuals via RAG.")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Technical question or fault description to retrieve evidence for."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rag.yaml",
        help="Path to RAG configuration YAML file."
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Maximum number of evidence chunks to retrieve."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional minimum similarity/relevance threshold."
    )
    parser.add_argument(
        "--equipment",
        type=str,
        default=None,
        help="Optional filter by equipment type (e.g., motor, pump, gearbox)."
    )
    parser.add_argument(
        "--format",
        choices=["structured", "json", "verbose"],
        default="structured",
        help="Output format."
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = RAGConfig.from_yaml(config_path)
    else:
        config = RAGConfig()

    embedding_model = create_embedding_model(config.embedding)
    vector_store = NumpyFlatVectorStore(config.vector_store)

    if vector_store.count() == 0:
        print("\n[ERROR] Vector store is empty! Please run:")
        print("  python scripts/ingest_documents.py --path data/rag/documents/\n")
        sys.exit(1)

    retriever = TechnicalRetriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
        config=config.retrieval,
    )

    filters = {}
    if args.equipment:
        filters["equipment_type"] = args.equipment

    evidence_items = retriever.retrieve(
        query=args.query,
        top_k=args.top_k,
        filters=filters if filters else None,
        similarity_threshold=args.threshold,
    )

    if not evidence_items:
        print("\nNo sufficiently relevant technical evidence found for query:")
        print(f"'{args.query}'\n")
        return

    print("\n=======================================================")
    print("         TECHNICAL KNOWLEDGE RETRIEVAL RESULTS         ")
    print("=======================================================")
    print(f"Query: \"{args.query}\"")
    print(f"Results Found: {len(evidence_items)} (Database chunks: {vector_store.count()})")
    print("=======================================================\n")

    for rank, ev in enumerate(evidence_items, start=1):
        print(f"--- [RESULT {rank}] ---")
        print(f"Document:     {ev.document_name}")
        print(f"Page:         {ev.page_number}")
        if ev.section:
            print(f"Section:      {ev.section}")
        if ev.equipment_type:
            print(f"Equipment:    {ev.equipment_type}")
        print(f"Score:        {ev.score:.4f} (mode: {ev.retrieval_mode})")
        print(f"Source Path:  {ev.source_path}")
        print("\nRelevant Text:")
        print(ev.text.strip())
        print("-" * 55 + "\n")


if __name__ == "__main__":
    main()
