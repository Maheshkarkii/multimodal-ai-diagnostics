"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { 
  CheckCircle2, 
  BookOpen, 
  Activity, 
  ArrowLeft,
  FileCheck,
  ShieldCheck,
  Sparkles,
  Layers,
  Clock,
  ArrowRight
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
      <div className="py-20 text-center space-y-4 max-w-md mx-auto">
        <div className="w-12 h-12 rounded-full bg-white border border-black/[0.06] flex items-center justify-center mx-auto text-[#1d1d1f] shadow-xs">
          <Activity className="w-6 h-6" />
        </div>
        <h2 className="text-xl font-semibold tracking-tight text-[#1d1d1f]">Report Not Available</h2>
        <p className="text-xs text-[#6e6e73] leading-relaxed">
          This session report has concluded or was initialized in another window.
        </p>
        <div className="pt-2">
          <Link
            href="/cases/new"
            className="inline-flex items-center space-x-2 bg-[#1d1d1f] hover:bg-black text-white px-5 py-2.5 rounded-full font-medium text-xs shadow-sm transition-all"
          >
            <span>Start New Case</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>
    );
  }

  const diag = diagnosisData.diagnosis;
  const decomp = diag.confidence_decomposition;

  return (
    <div className="space-y-8 max-w-4xl mx-auto">
      {/* Header & Back Link */}
      <div className="flex items-center justify-between">
        <Link
          href="/"
          className="text-xs font-medium text-[#6e6e73] hover:text-[#1d1d1f] transition-colors flex items-center space-x-1.5 px-3 py-1.5 rounded-full hover:bg-black/[0.03]"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Dashboard</span>
        </Link>
        <div className="text-[11px] text-[#86868b] flex items-center space-x-1.5">
          <span>Case:</span>
          <span className="font-mono font-medium text-[#1d1d1f] bg-black/[0.04] px-2 py-0.5 rounded-full border border-black/[0.04]">
            {diagnosisData.case_id}
          </span>
        </div>
      </div>

      {/* Primary Diagnosis Hero Banner */}
      <div className="bg-white rounded-[28px] border border-black/[0.06] p-8 sm:p-10 shadow-[0_4px_24px_rgba(0,0,0,0.03)] space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-black/[0.04]">
          <div className="space-y-1">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-[#86868b]">
              Primary Diagnostic Assessment
            </span>
            <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-[#1d1d1f] capitalize">
              {diag.primary_diagnosis.replace(/_/g, " ")}
            </h1>
          </div>
          <div className="flex items-center space-x-2.5 self-start sm:self-center">
            <span className="px-3.5 py-1 rounded-full text-xs font-medium bg-[#f5f5f7] text-[#1d1d1f] border border-black/[0.06]">
              Severity: {diag.severity}
            </span>
            <span className="px-3.5 py-1 rounded-full text-xs font-medium bg-[#1d1d1f] text-white shadow-xs">
              {(diag.diagnostic_confidence * 100).toFixed(1)}% Confidence
            </span>
          </div>
        </div>

        <div className="p-5 bg-[#f5f5f7] rounded-[18px] border border-black/[0.04] text-sm text-[#1d1d1f] leading-relaxed">
          <span className="font-semibold">Symptom Summary: </span>
          <span className="text-[#515154]">{diagnosisData.problem_summary}</span>
        </div>
      </div>

      {/* Confidence Decomposition Breakdown */}
      {decomp && (
        <div className="bg-white rounded-[24px] border border-black/[0.06] p-6 sm:p-8 shadow-[0_2px_14px_rgba(0,0,0,0.02)] space-y-5">
          <div className="flex items-center justify-between pb-3 border-b border-black/[0.04]">
            <h2 className="text-base font-semibold text-[#1d1d1f] flex items-center space-x-2">
              <Activity className="w-4 h-4 text-[#1d1d1f]" />
              <span>Confidence Decomposition & Subsystem Attribution</span>
            </h2>
            <span className="text-[11px] text-[#86868b]">Cross-modal verification</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5 text-xs">
            <div className="p-4 bg-[#f5f5f7] rounded-[16px] border border-black/[0.04] space-y-1">
              <div className="text-[#86868b] text-[11px]">Modality Agreement</div>
              <div className="text-sm font-semibold text-[#1d1d1f] uppercase tracking-wide">
                {decomp.multimodal_agreement}
              </div>
            </div>
            <div className="p-4 bg-[#f5f5f7] rounded-[16px] border border-black/[0.04] space-y-1">
              <div className="text-[#86868b] text-[11px]">Sensor Evidence</div>
              <div className="text-sm font-semibold text-[#1d1d1f] uppercase tracking-wide">
                {decomp.sensor_evidence_strength}
              </div>
            </div>
            <div className="p-4 bg-[#f5f5f7] rounded-[16px] border border-black/[0.04] space-y-1">
              <div className="text-[#86868b] text-[11px]">Acoustic Harmonics</div>
              <div className="text-sm font-semibold text-[#1d1d1f] uppercase tracking-wide">
                {decomp.acoustic_evidence_strength}
              </div>
            </div>
            <div className="p-4 bg-[#f5f5f7] rounded-[16px] border border-black/[0.04] space-y-1">
              <div className="text-[#86868b] text-[11px]">Knowledge Base Match</div>
              <div className="text-sm font-semibold text-[#1d1d1f] uppercase tracking-wide">
                {decomp.technical_knowledge_match}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Evidence Section */}
      <div className="bg-white rounded-[24px] border border-black/[0.06] p-6 sm:p-8 shadow-[0_2px_14px_rgba(0,0,0,0.02)] space-y-5">
        <div className="flex items-center justify-between pb-3 border-b border-black/[0.04]">
          <h2 className="text-base font-semibold text-[#1d1d1f] flex items-center space-x-2">
            <BookOpen className="w-4 h-4 text-[#1d1d1f]" />
            <span>Auditable Evidence Trail</span>
          </h2>
          <span className="text-[11px] text-[#86868b]">Immutable citations</span>
        </div>

        <div className="space-y-3">
          {diagnosisData.evidence_inventory.map((ev) => (
            <div
              key={ev.evidence_id}
              className="p-4 rounded-[18px] border border-black/[0.04] bg-[#f5f5f7] hover:bg-[#fafafc] transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-sm"
            >
              <div className="space-y-1.5">
                <div className="flex items-center space-x-2">
                  <span className="font-mono text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-white text-[#1d1d1f] border border-black/[0.06] shadow-2xs">
                    {ev.evidence_id}
                  </span>
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-[#6e6e73]">
                    {ev.category}
                  </span>
                  {ev.is_verified_citation && (
                    <span className="text-[10px] bg-[#34c759]/10 text-[#34c759] border border-[#34c759]/20 font-medium px-2 py-0.5 rounded-full">
                      Verified Citation
                    </span>
                  )}
                </div>
                <p className="text-[#1d1d1f] text-xs sm:text-[13px] leading-relaxed">{ev.description}</p>
                {ev.document_reference && (
                  <p className="text-[11px] text-[#86868b]">
                    Reference: <span className="text-[#515154] font-medium">{ev.document_reference}</span> (p. {ev.page_number || "N/A"})
                  </p>
                )}
              </div>
              <div className="text-xs text-[#6e6e73] font-medium self-end sm:self-center bg-white px-2.5 py-1 rounded-full border border-black/[0.04] shadow-2xs">
                {(ev.confidence * 100).toFixed(0)}% Confidence
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recommended Actions */}
      <div className="bg-white rounded-[24px] border border-black/[0.06] p-6 sm:p-8 shadow-[0_2px_14px_rgba(0,0,0,0.02)] space-y-5">
        <div className="flex items-center justify-between pb-3 border-b border-black/[0.04]">
          <h2 className="text-base font-semibold text-[#1d1d1f] flex items-center space-x-2">
            <FileCheck className="w-4 h-4 text-[#1d1d1f]" />
            <span>Recommended Corrective Actions</span>
          </h2>
          <span className="text-[11px] text-[#86868b]">SOP procedures</span>
        </div>

        <div className="space-y-3">
          {diagnosisData.recommended_actions.map((act) => (
            <div
              key={act.action_id}
              className="p-5 rounded-[18px] border border-black/[0.06] bg-[#fbfbfd] space-y-2 text-sm"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="font-semibold text-[#1d1d1f]">{act.description}</span>
                <span className="text-[10px] font-medium px-2.5 py-0.5 rounded-full uppercase bg-[#f5f5f7] text-[#6e6e73] border border-black/[0.06] whitespace-nowrap">
                  {act.action_type.replace(/_/g, " ")}
                </span>
              </div>
              <p className="text-xs text-[#6e6e73]">Target Component: <span className="text-[#1d1d1f] font-medium">{act.target_component}</span></p>
              {act.citation && (
                <div className="text-[11px] text-[#86868b] pt-1">
                  Standard SOP Citation: <span className="font-mono text-[#515154]">{act.citation}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Human Feedback Section */}
      <div className="bg-white rounded-[24px] border border-black/[0.06] p-6 sm:p-8 shadow-[0_2px_14px_rgba(0,0,0,0.02)] space-y-4">
        <div>
          <h2 className="text-base font-semibold text-[#1d1d1f]">Technician Feedback & Verification</h2>
          <p className="text-xs text-[#6e6e73] mt-0.5">
            Validate this assessment to calibrate future multimodal models.
          </p>
        </div>

        {feedbackSubmitted ? (
          <div className="p-4 bg-[#34c759]/10 border border-[#34c759]/20 text-[#34c759] text-xs font-medium rounded-[16px] flex items-center space-x-2">
            <CheckCircle2 className="w-4 h-4 text-[#34c759]" />
            <span>Thank you. Your feedback has been recorded for continuous model calibration.</span>
          </div>
        ) : (
          <div className="flex flex-wrap items-center gap-3 pt-1">
            <button
              type="button"
              onClick={() => setFeedbackSubmitted(true)}
              className="px-5 py-2.5 bg-[#1d1d1f] hover:bg-black text-white text-xs font-medium rounded-full shadow-xs transition-all active:scale-95"
            >
              Accurate Diagnosis
            </button>
            <button
              type="button"
              onClick={() => setFeedbackSubmitted(true)}
              className="px-5 py-2.5 bg-[#f5f5f7] hover:bg-[#e8e8ed] text-[#1d1d1f] border border-black/[0.06] text-xs font-medium rounded-full transition-all active:scale-95"
            >
              Partially Accurate / Needs Correction
            </button>
          </div>
        )}
      </div>
    </div>
  );
}