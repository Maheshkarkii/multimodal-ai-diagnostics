"""
Comprehensive Unit & Integration Test Suite for Phase 11.
Tests Classification Metrics, Calibration (ECE, Brier), Anomaly Detection, RAG Citations,
Modality Ablation, Regression Gate, Production Monitoring, and Human Feedback.
"""

from src.evaluation.benchmarks.benchmark_runner import BenchmarkRunner
from src.evaluation.calibration.calibration import compute_calibration_metrics
from src.evaluation.evaluators.anomaly import evaluate_anomaly_detector
from src.evaluation.evaluators.rag_evidence import evaluate_rag_and_citations
from src.evaluation.metrics.classification import calculate_classification_metrics
from src.evaluation.regression.gate import check_regression_against_baseline
from src.evaluation.reporting.report_generator import generate_markdown_evaluation_report
from src.evaluation.robustness.ablation import run_modality_ablation_study
from src.feedback.schemas import FeedbackCategory, HumanDiagnosticFeedback
from src.feedback.storage import FeedbackStore
from src.monitoring.events import DiagnosticMonitoringEvent
from src.monitoring.storage import MonitoringStore


def test_classification_metrics_calculation():
    """Verify multi-class precision, recall, macro/weighted F1, and confusion matrix."""
    y_true = ["Bearing", "Bearing", "Misalignment", "Normal"]
    y_pred = ["Bearing", "Normal", "Misalignment", "Normal"]

    metrics = calculate_classification_metrics(y_true, y_pred)
    assert metrics.sample_count == 4
    assert metrics.accuracy == 0.75
    assert 0.0 <= metrics.f1_macro <= 1.0
    assert "Bearing" in metrics.per_class_f1
    assert len(metrics.confusion_matrix) == 3


def test_confidence_calibration_ece_and_brier():
    """Verify Expected Calibration Error (ECE) and Brier Score computations."""
    y_true = [1, 1, 0, 1, 0]
    y_pred = [1, 1, 1, 1, 0]
    y_prob = [0.95, 0.85, 0.90, 0.70, 0.80]

    cal = compute_calibration_metrics(y_true, y_prob, y_pred, num_bins=5)
    assert 0.0 <= cal.expected_calibration_error <= 1.0
    assert 0.0 <= cal.brier_score <= 1.0
    assert len(cal.bin_confidences) == 5


def test_anomaly_detection_evaluator():
    """Verify binary anomaly detection precision, recall, and false alarm rate."""
    y_t = [True, True, False, False]
    y_p = [True, False, False, False]  # 1 TP, 1 FN, 2 TN, 0 FP

    res = evaluate_anomaly_detector(y_t, y_p)
    assert res.precision == 1.0
    assert res.recall == 0.5
    assert res.false_alarm_rate == 0.0
    assert res.missed_anomaly_rate == 0.5


def test_rag_retrieval_and_citation_evaluator():
    """Verify HitRate@1, MRR, nDCG, and Unsupported Claim Rate."""
    queries = [
        {"target_doc_id": "DOC-A", "retrieved_doc_ids": ["DOC-A", "DOC-B"]},
        {"target_doc_id": "DOC-B", "retrieved_doc_ids": ["DOC-C", "DOC-B"]},
    ]
    claims = [
        {"claim": "Bearing overtemp", "is_supported": True, "citation": "DOC-A", "has_valid_citation": True},
        {"claim": "Invented claim", "is_supported": False},
    ]

    res = evaluate_rag_and_citations(queries, claims)
    assert res.hit_rate_at_1 == 0.5
    assert res.hit_rate_at_5 == 1.0
    assert res.mrr == 0.75
    assert res.unsupported_claim_rate == 0.5


def test_modality_ablation_and_robustness():
    """Verify leave-one-modality-out degradation profiling."""
    cases = [
        {"ground_truth_fault": "Bearing Fault"},
        {"ground_truth_fault": "Misalignment"},
    ]

    def mock_predict(case, mask):
        if not mask.get("sensor", True):
            return "Uncertain"
        return case["ground_truth_fault"]

    rob = run_modality_ablation_study(cases, mock_predict)
    assert rob.full_modality_f1 == 1.0
    assert rob.missing_sensor_f1 == 0.0
    assert rob.abstention_rate_on_insufficient_data > 0.0


def test_regression_gate_pass_and_fail():
    """Verify CI regression gate rejects metric drops exceeding thresholds."""
    runner = BenchmarkRunner()
    baseline = runner.run_full_evaluation()
    candidate_pass = runner.run_full_evaluation()

    passed, details = check_regression_against_baseline(candidate_pass, baseline)
    assert passed is True
    assert details["gate_passed"] is True

    # Artificially regress candidate metrics
    candidate_fail = runner.run_full_evaluation()
    candidate_fail.fusion_metrics.f1_macro = 0.50  # Heavy drop
    passed_fail, details_fail = check_regression_against_baseline(candidate_fail, baseline)
    assert passed_fail is False
    assert details_fail.get("fusion_f1_regressed") is True


def test_monitoring_telemetry_storage(tmp_path):
    """Verify thread-safe monitoring telemetry persistence and retrieval."""
    log_file = tmp_path / "events.jsonl"
    store = MonitoringStore(str(log_file))

    evt = DiagnosticMonitoringEvent(
        event_id="EVT-001",
        request_id="REQ-123",
        case_id="CASE-456",
        primary_diagnosis="Bearing Degradation",
        confidence=0.92,
        pipeline_latency_ms=75.4,
    )
    store.record_event(evt)

    events = store.load_events()
    assert len(events) == 1
    assert events[0].event_id == "EVT-001"
    assert events[0].pipeline_latency_ms == 75.4


def test_human_feedback_lifecycle_and_analysis(tmp_path):
    """Verify human feedback submission, persistence, and disagreement aggregation."""
    log_file = tmp_path / "feedback.jsonl"
    store = FeedbackStore(str(log_file))

    fb1 = HumanDiagnosticFeedback(
        feedback_id="FB-001",
        case_id="CASE-456",
        reviewer_id="ENG-07",
        category=FeedbackCategory.CORRECT,
        is_diagnosis_accurate=True,
    )
    fb2 = HumanDiagnosticFeedback(
        feedback_id="FB-002",
        case_id="CASE-789",
        reviewer_id="ENG-07",
        category=FeedbackCategory.INCORRECT,
        is_diagnosis_accurate=False,
        ground_truth_correction="Shaft Misalignment",
    )
    store.submit_feedback(fb1)
    store.submit_feedback(fb2)

    analysis = store.analyze_feedback()
    assert analysis["total_feedback_count"] == 2
    assert analysis["accuracy_rate"] == 0.5
    assert analysis["frequent_corrections"]["Shaft Misalignment"] == 1


def test_markdown_evaluation_report_generation(tmp_path):
    """Verify automated markdown report synthesis."""
    runner = BenchmarkRunner()
    summary = runner.run_full_evaluation()
    out_md = tmp_path / "eval_report.md"

    content = generate_markdown_evaluation_report(summary, str(out_md))
    assert out_md.exists()
    assert "AI Field Engineer" in content
    assert summary.evaluation_id in content
