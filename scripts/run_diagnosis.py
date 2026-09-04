"""
CLI Script: Run Autonomous Multimodal Diagnostic Reasoning Agent on a field case.

Usage:
    python scripts/run_diagnosis.py --equipment motor --description "High pitch acoustic squealing and excessive vibration"
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.core.agent import DiagnosticReasoningAgent
from src.agent.core.schema import ModalityObservation, ModalityType
from src.rag.config import RAGConfig
from src.rag.embeddings.model import create_embedding_model
from src.rag.vectorstore.store import NumpyFlatVectorStore
from src.rag.retrieval.retriever import TechnicalRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("run_diagnosis")


def main():
    parser = argparse.ArgumentParser(description="Run autonomous diagnostic reasoning workflow.")
    parser.add_argument("--equipment", type=str, default="motor", help="Equipment type (e.g. motor, pump, gearbox)")
    parser.add_argument("--model", type=str, default=None, help="Equipment model name (e.g. M-4500, CP-800)")
    parser.add_argument("--description", "-d", type=str, required=True, help="Technician symptom description")
    parser.add_argument("--vibration", type=float, default=None, help="Observed vibration RMS velocity (mm/s)")
    parser.add_argument("--temp", type=float, default=None, help="Observed temperature (deg C)")
    parser.add_argument("--vision-pred", type=str, default=None, help="Optional vision model prediction")
    parser.add_argument("--audio-pred", type=str, default=None, help="Optional audio model prediction")
    parser.add_argument("--config", type=str, default="configs/rag.yaml", help="RAG config path")
    parser.add_argument("--out", type=str, default=None, help="Path to save output markdown report")

    args = parser.parse_args()

    # Initialize RAG retrieval backend
    rag_cfg = RAGConfig.from_yaml(args.config) if Path(args.config).exists() else RAGConfig()
    emb_model = create_embedding_model(rag_cfg.embedding)
    store = NumpyFlatVectorStore(rag_cfg.vector_store)

    if store.count() == 0:
        logger.error("RAG vector store is empty! Please run 'python scripts/ingest_documents.py' first.")
        sys.exit(1)

    retriever = TechnicalRetriever(store, emb_model, rag_cfg.retrieval)
    agent = DiagnosticReasoningAgent(retriever)

    # Prepare observations
    observations = {}
    if args.vision_pred:
        observations["vision"] = ModalityObservation(
            modality=ModalityType.VISION,
            prediction=args.vision_pred,
            confidence=0.85,
        )
    if args.audio_pred:
        observations["audio"] = ModalityObservation(
            modality=ModalityType.AUDIO,
            prediction=args.audio_pred,
            confidence=0.88,
        )

    # Prepare sensor telemetry
    sensor_data = []
    if args.vibration is not None:
        sensor_data.append({
            "parameter": "Vibration_RMS",
            "value": args.vibration,
            "unit": "mm/s",
            "warning_threshold": 4.5,
            "critical_threshold": 7.1,
        })
    if args.temp is not None:
        sensor_data.append({
            "parameter": "Temperature",
            "value": args.temp,
            "unit": "degC",
            "warning_threshold": 75.0,
            "critical_threshold": 90.0,
        })

    # Execute Autonomous Diagnostic Reasoning
    logger.info("Executing Autonomous Diagnostic Reasoning Workflow...")
    report = agent.diagnose_case(
        equipment_type=args.equipment,
        equipment_model=args.model,
        technician_description=args.description,
        observations=observations if observations else None,
        sensor_data=sensor_data if sensor_data else None,
    )

    markdown_report = report.to_markdown()
    print("\n" + markdown_report + "\n")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown_report, encoding="utf-8")
        logger.info(f"Saved report to '{out_path}'")


if __name__ == "__main__":
    main()
