"""
CLI Script: Generate Auditable Diagnostic & Explainability Report.

Usage:
    python scripts/generate_auditable_report.py --equipment motor --model M-4500 --description "High pitch acoustic squealing from bearing" --vibration 6.8 --temp 84.0 --audio-pred "bearing_defect_wear"
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
from src.explainability.core.config import ExplainabilityConfig
from src.explainability.core.service import ExplainabilityService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("generate_auditable_report")


def main():
    parser = argparse.ArgumentParser(description="Generate full auditable diagnostic report with evidence attribution.")
    parser.add_argument("--equipment", type=str, default="motor", help="Equipment type")
    parser.add_argument("--model", type=str, default=None, help="Equipment model")
    parser.add_argument("--description", "-d", type=str, required=True, help="Technician problem description")
    parser.add_argument("--vibration", type=float, default=None, help="Vibration RMS velocity (mm/s)")
    parser.add_argument("--temp", type=float, default=None, help="Temperature (deg C)")
    parser.add_argument("--audio-pred", type=str, default=None, help="Audio model prediction")
    parser.add_argument("--vision-pred", type=str, default=None, help="Vision model prediction")
    parser.add_argument("--out", type=str, default=None, help="Optional output markdown report path")

    args = parser.parse_args()

    # Initialize RAG & Agent
    rag_cfg = RAGConfig()
    emb_model = create_embedding_model(rag_cfg.embedding)
    store = NumpyFlatVectorStore(rag_cfg.vector_store)

    if store.count() == 0:
        logger.error("RAG store empty. Please run 'python scripts/ingest_documents.py' first.")
        sys.exit(1)

    retriever = TechnicalRetriever(store, emb_model, rag_cfg.retrieval)
    agent = DiagnosticReasoningAgent(retriever)
    explainability_service = ExplainabilityService(ExplainabilityConfig())

    # Formulate observations
    obs = {}
    if args.audio_pred:
        obs["audio"] = ModalityObservation(
            modality=ModalityType.AUDIO,
            prediction=args.audio_pred,
            confidence=0.88,
        )
    if args.vision_pred:
        obs["vision"] = ModalityObservation(
            modality=ModalityType.VISION,
            prediction=args.vision_pred,
            confidence=0.85,
        )

    # Formulate sensor telemetry
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
            "parameter": "Bearing_Temp",
            "value": args.temp,
            "unit": "degC",
            "warning_threshold": 75.0,
            "critical_threshold": 90.0,
        })

    logger.info("Executing Phase 7 Diagnostic Reasoning Agent...")
    diag_report = agent.diagnose_case(
        equipment_type=args.equipment,
        equipment_model=args.model,
        technician_description=args.description,
        observations=obs if obs else None,
        sensor_data=sensor_data if sensor_data else None,
    )

    logger.info("Synthesizing Phase 8 Auditable Evidence & Explainability Layer...")
    auditable_report = explainability_service.generate_auditable_report(
        diagnostic_report=diag_report,
        raw_inputs={"description": args.description, "vibration": args.vibration, "temp": args.temp},
    )

    markdown_output = auditable_report.to_markdown()
    print("\n" + markdown_output + "\n")

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(markdown_output, encoding="utf-8")
        logger.info(f"Auditable report saved to '{out_p}'")


if __name__ == "__main__":
    main()
