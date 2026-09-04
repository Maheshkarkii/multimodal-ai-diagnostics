"""
Evaluation dataset and runner for RAG Technical Knowledge Retrieval.
Contains ground-truth annotated benchmark queries across pump, motor, and gearbox failure modes.
"""

from pathlib import Path
import json
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.config import RAGConfig
from src.rag.embeddings.model import create_embedding_model
from src.rag.vectorstore.store import NumpyFlatVectorStore
from src.rag.retrieval.retriever import TechnicalRetriever
from src.rag.evaluation.evaluator import EvaluationSample, RAGEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("evaluate_rag")

BENCHMARK_SAMPLES = [
    EvaluationSample(
        query_id="Q01_BEARING_INSPECTION",
        query="What are the recommended inspection steps for bearing problems?",
        target_document_name="motor_m4500_maintenance_manual.pdf",
        target_page=2,
        target_section="BEARING INSPECTION",
        target_keywords=["ultrasound", "RMS vibration", "thermal infrared", "grease sample", "radial play"],
        equipment_type="motor",
        description="Verify bearing inspection procedures from motor manual page 2."
    ),
    EvaluationSample(
        query_id="Q02_VIBRATION_SEVERITY",
        query="What vibration level requires immediate inspection or shutdown according to ISO standard?",
        target_document_name="motor_m4500_maintenance_manual.pdf",
        target_page=1,
        target_section="VIBRATION THRESHOLDS",
        target_keywords=["ISO 10816-3", "Zone C", "Zone D", "7.1 mm/s", "emergency shutdown"],
        equipment_type="motor",
        description="Check ISO 10816-3 threshold retrieval from motor manual page 1."
    ),
    EvaluationSample(
        query_id="Q03_CAVITATION_SYMPTOMS",
        query="What are the symptoms and acoustic noise associated with hydraulic cavitation?",
        target_document_name="centrifugal_pump_cp800_troubleshooting.md",
        target_page=1,
        target_section="CAVITATION",
        target_keywords=["gravel-like popping", "5 kHz", "15 kHz", "suction pressure", "impeller"],
        equipment_type="pump",
        description="Retrieve hydraulic cavitation acoustics and symptoms from pump guide."
    ),
    EvaluationSample(
        query_id="Q04_MECHANICAL_SEAL_LEAKAGE",
        query="What is the allowable leakage rate for centrifugal pump mechanical seals?",
        target_document_name="centrifugal_pump_cp800_troubleshooting.md",
        target_page=2,
        target_section="MECHANICAL SEAL",
        target_keywords=["drops per minute", "10 ml/hour", "carbon", "silicon-carbide"],
        equipment_type="pump",
        description="Retrieve mechanical seal leakage tolerances from pump guide page 2."
    ),
    EvaluationSample(
        query_id="Q05_GEARBOX_PITTING_NOISE",
        query="What causes gear mesh frequency sidebands and high oil iron content in gearboxes?",
        target_document_name="industrial_gearbox_gb200_repair.txt",
        target_page=1,
        target_section="GEAR MESHING",
        target_keywords=["pitting", "gear mesh frequency", "GMF", "Iron", "100 ppm"],
        equipment_type="gearbox",
        description="Retrieve gear tooth pitting and oil contamination limits from gearbox repair guide."
    ),
    EvaluationSample(
        query_id="Q06_ROTATING_UNBALANCE_HARMONIC",
        query="What vibration frequency indicates rotor dynamic unbalance and when should balancing be performed?",
        target_document_name="motor_m4500_maintenance_manual.pdf",
        target_page=3,
        target_section="ROTOR UNBALANCE",
        target_keywords=["1X running speed", "5.0 mm/s", "dynamic two-plane balancing"],
        equipment_type="motor",
        description="Retrieve rotor unbalance 1X harmonics from motor manual page 3."
    )
]


def run_evaluation(config_path: str = "configs/rag.yaml", top_k: int = 5):
    config = RAGConfig.from_yaml(config_path) if Path(config_path).exists() else RAGConfig()
    embedding_model = create_embedding_model(config.embedding)
    vector_store = NumpyFlatVectorStore(config.vector_store)

    if vector_store.count() == 0:
        logger.error("Vector store empty. Please run 'python scripts/ingest_documents.py' first.")
        return

    retriever = TechnicalRetriever(vector_store, embedding_model, config.retrieval)
    evaluator = RAGEvaluator(retriever)

    print("\n=======================================================")
    print("      TECHNICAL KNOWLEDGE RAG BENCHMARK EVALUATION     ")
    print("=======================================================")
    print(f"Total Benchmark Queries: {len(BENCHMARK_SAMPLES)}")
    print(f"Retrieval Depth (top-k): {top_k}")
    print("-------------------------------------------------------")

    metrics = evaluator.evaluate_benchmark(BENCHMARK_SAMPLES, top_k=top_k)

    print(f"Hit Rate @ 1 (Top-1 Accuracy):  {metrics.hit_rate_at_1 * 100:.2f}%")
    print(f"Hit Rate @ 3:                   {metrics.hit_rate_at_3 * 100:.2f}%")
    print(f"Hit Rate @ 5:                   {metrics.hit_rate_at_5 * 100:.2f}%")
    print(f"Mean Reciprocal Rank (MRR):     {metrics.mrr:.4f}")
    print(f"Precision @ {top_k}:                {metrics.precision_at_k:.4f}")
    print(f"Recall @ {top_k}:                   {metrics.recall_at_k:.4f}")
    print(f"Average Top-1 Score:            {metrics.average_top1_score:.4f}")
    print("=======================================================\n")

    print("Detailed Query Breakdown:")
    for res in metrics.per_query_results:
        status = "PASSED" if res["matched_rank"] is not None and res["matched_rank"] <= 3 else "FAILED"
        rank_str = f"Rank #{res['matched_rank']}" if res["matched_rank"] is not None else "NOT FOUND"
        print(f"[{status}] {res['query_id']}: {rank_str} (Score: {res['top_score']:.4f})")
        print(f"   Target: {res['target_doc']} (P.{res['target_page']}) | Top Retrieved: {res['top_result_doc']} (P.{res['top_result_page']})")

    # Save metrics report
    report_path = Path("reports/rag_retrieval_benchmark.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_queries": metrics.total_queries,
            "hit_rate_at_1": metrics.hit_rate_at_1,
            "hit_rate_at_3": metrics.hit_rate_at_3,
            "hit_rate_at_5": metrics.hit_rate_at_5,
            "mrr": metrics.mrr,
            "precision_at_k": metrics.precision_at_k,
            "recall_at_k": metrics.recall_at_k,
            "average_top1_score": metrics.average_top1_score,
            "per_query_results": metrics.per_query_results
        }, f, indent=2)
    print(f"\nSaved evaluation metrics report to '{report_path}'\n")


if __name__ == "__main__":
    run_evaluation()
