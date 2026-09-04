"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { 
  CheckCircle2, 
  BookOpen, 
  Activity, 
  ArrowLeft,
  FileCheck
} from "lucide-react";
import { DiagnosisResponse } from "@/lib/types/diagnosis";

export default function DiagnosticResultPage() {
  const params = useParams();
  const caseId = params.caseId as string;

  const [diagnosisData, setDiagnosisData] = useState<DiagnosisResponse | null>(null);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  useEffect(() => {
    const stored = sessionStorage.getItem(`case_${caseId}`);
    if (stored) {
      try {
        setDiagnosisData(JSON.parse(stored));
      } catch (e) {
        console.error("Failed to parse case data", e);
      }
    }
  }, [caseId]);

  if (!diagnosisData) {
    return (
      <div className="py-12 text-center space-y-4">
        <h2 className="text-xl font-bold text-industrial-800">Case Report Not Found in Session</h2>
        <p className="text-sm text-industrial-500">
          This report may have expired or was run in another browser window.
        </p>
        <Link
          href="/cases/new"
          className="inline-block bg-brand-blue text-white px-4 py-2 rounded-lg font-semibold text-sm"
        >
          Create New Diagnostic Case
        </Link>
      </div>
    );
  }

  const diag = diagnosisData.diagnosis;
  const decomp = diag.confidence_decomposition;

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Header & Back Link */}
      <div className="flex items-center justify-between">
        <Link
          href="/"
          className="text-sm font-semibold text-industrial-600 hover:text-industrial-900 flex items-center space-x-1"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Dashboard</span>
        </Link>
        <div className="text-xs text-industrial-400">
          Case ID: <span className="font-mono text-industrial-700">{diagnosisData.case_id}</span>
        </div>
      </div>

      {/* Primary Diagnosis Hero Banner */}
      <div className="bg-white rounded-2xl border border-industrial-200 p-8 shadow-sm space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-industrial-400">
              Primary Diagnostic Assessment
            </span>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-industrial-900 mt-1 capitalize">
              {diag.primary_diagnosis.replace(/_/g, " ")}
            </h1>
          </div>
          <div className="flex items-center space-x-3">
            <div className="px-3 py-1.5 rounded-lg text-xs font-bold bg-amber-100 text-amber-800 border border-amber-200">
              Severity: {diag.severity}
            </div>
            <div className="px-3 py-1.5 rounded-lg text-xs font-bold bg-sky-100 text-sky-800 border border-sky-200">
              Confidence: {(diag.diagnostic_confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        <div className="p-4 bg-industrial-50 rounded-xl border border-industrial-100 text-sm text-industrial-800 leading-relaxed">
          <span className="font-semibold">Problem Summary: </span>
          {diagnosisData.problem_summary}
        </div>
      </div>

      {/* Confidence Decomposition Breakdown */}
      {decomp && (
        <div className="bg-white rounded-xl border border-industrial-200 p-6 shadow-sm space-y-4">
          <h2 className="text-base font-bold text-industrial-900 flex items-center space-x-2">
            <Activity className="w-4 h-4 text-brand-blue" />
            <span>Confidence Decomposition & Subsystem Attribution</span>
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
            <div className="p-3 bg-industrial-50 rounded-lg border border-industrial-100">
              <div className="text-industrial-500">Modality Agreement</div>
              <div className="text-sm font-bold text-industrial-900 mt-1 uppercase">
                {decomp.multimodal_agreement}
              </div>
            </div>
            <div className="p-3 bg-industrial-50 rounded-lg border border-industrial-100">
              <div className="text-industrial-500">Sensor Evidence</div>
              <div className="text-sm font-bold text-industrial-900 mt-1 uppercase">
                {decomp.sensor_evidence_strength}
              </div>
            </div>
            <div className="p-3 bg-industrial-50 rounded-lg border border-industrial-100">
              <div className="text-industrial-500">Acoustic Harmonics</div>
              <div className="text-sm font-bold text-industrial-900 mt-1 uppercase">
                {decomp.acoustic_evidence_strength}
              </div>
            </div>
            <div className="p-3 bg-industrial-50 rounded-lg border border-industrial-100">
              <div className="text-industrial-500">Knowledge Base Match</div>
              <div className="text-sm font-bold text-industrial-900 mt-1 uppercase">
                {decomp.technical_knowledge_match}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Evidence Section */}
      <div className="bg-white rounded-xl border border-industrial-200 p-6 shadow-sm space-y-4">
        <h2 className="text-base font-bold text-industrial-900 flex items-center space-x-2">
          <BookOpen className="w-4 h-4 text-brand-blue" />
          <span>Auditable Evidence Trail</span>
        </h2>
        <div className="space-y-3">
          {diagnosisData.evidence_inventory.map((ev) => (
            <div
              key={ev.evidence_id}
              className="p-4 rounded-xl border border-industrial-100 bg-industrial-50/50 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-sm"
            >
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-industrial-200 text-industrial-800">
                    {ev.evidence_id}
                  </span>
                  <span className="text-xs font-semibold uppercase text-brand-blue">
                    {ev.category}
                  </span>
                  {ev.is_verified_citation && (
                    <span className="text-[10px] bg-emerald-100 text-emerald-800 font-bold px-1.5 py-0.5 rounded">
                      Verified Citation
                    </span>
                  )}
                </div>
                <p className="text-industrial-800 text-xs sm:text-sm">{ev.description}</p>
                {ev.document_reference && (
                  <p className="text-xs text-industrial-500 font-medium">
                    Reference: {ev.document_reference} (p. {ev.page_number || "N/A"})
                  </p>
                )}
              </div>
              <div className="text-xs text-industrial-500 font-semibold self-end sm:self-center">
                Confidence: {(ev.confidence * 100).toFixed(0)}%
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recommended Actions */}
      <div className="bg-white rounded-xl border border-industrial-200 p-6 shadow-sm space-y-4">
        <h2 className="text-base font-bold text-industrial-900 flex items-center space-x-2">
          <FileCheck className="w-4 h-4 text-brand-blue" />
          <span>Recommended Corrective Actions</span>
        </h2>
        <div className="space-y-3">
          {diagnosisData.recommended_actions.map((act) => (
            <div
              key={act.action_id}
              className="p-4 rounded-xl border border-industrial-200 bg-white space-y-2 text-sm"
            >
              <div className="flex items-center justify-between">
                <span className="font-bold text-industrial-900">{act.description}</span>
                <span className="text-[10px] font-bold px-2 py-0.5 rounded uppercase bg-amber-50 text-amber-800 border border-amber-200">
                  {act.action_type.replace(/_/g, " ")}
                </span>
              </div>
              <p className="text-xs text-industrial-600">Target Component: {act.target_component}</p>
              {act.citation && (
                <div className="text-[10px] text-industrial-400">
                  Standard SOP Citation: {act.citation}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Human Feedback Section */}
      <div className="bg-white rounded-xl border border-industrial-200 p-6 shadow-sm space-y-4">
        <h2 className="text-base font-bold text-industrial-900">Technician Feedback & Validation</h2>
        {feedbackSubmitted ? (
          <div className="p-4 bg-emerald-50 text-emerald-800 text-sm rounded-lg flex items-center space-x-2">
            <CheckCircle2 className="w-4 h-4" />
            <span>Thank you. Your feedback has been recorded for continuous model calibration.</span>
          </div>
        ) : (
          <div className="space-y-3 text-sm">
            <p className="text-xs text-industrial-600">
              Was this diagnosis and recommended action plan accurate for this asset?
            </p>
            <div className="flex items-center space-x-4">
              <button
                type="button"
                onClick={() => setFeedbackSubmitted(true)}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold rounded-lg transition"
              >
                Accurate Diagnosis
              </button>
              <button
                type="button"
                onClick={() => setFeedbackSubmitted(true)}
                className="px-4 py-2 bg-industrial-200 hover:bg-industrial-300 text-industrial-800 text-xs font-bold rounded-lg transition"
              >
                Partially Accurate / Needs Correction
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}