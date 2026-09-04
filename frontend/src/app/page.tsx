"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { 
  Activity, 
  Wrench, 
  CheckCircle2, 
  AlertCircle, 
  Layers, 
  BookOpen, 
  ShieldCheck, 
  ArrowRight,
  Server
} from "lucide-react";
import { apiClient } from "@/lib/api/client";
import { ReadinessResponse, HealthResponse } from "@/lib/types/diagnosis";

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function checkStatus() {
      try {
        const [h, r] = await Promise.all([
          apiClient.checkHealth().catch(() => null),
          apiClient.checkReadiness().catch(() => null),
        ]);
        setHealth(h);
        setReadiness(r);
      } catch (e) {
        console.error("Status check failed", e);
      } finally {
        setLoading(false);
      }
    }
    checkStatus();
  }, []);

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-industrial-900 to-industrial-800 rounded-2xl p-8 text-white shadow-md border border-industrial-700">
        <div className="max-w-3xl space-y-4">
          <div className="inline-flex items-center space-x-2 bg-brand-blue/20 text-sky-300 border border-sky-500/30 px-3 py-1 rounded-full text-xs font-semibold">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Industrial Autonomous Troubleshooting</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight">
            AI Field Engineer
          </h1>
          <p className="text-industrial-300 text-sm sm:text-base leading-relaxed">
            Multi-modal diagnostic reasoning combining equipment vision saliency, acoustic spectrum harmonics, physical telemetry thresholds, and verified OEM technical knowledge retrieval.
          </p>
          <div className="pt-2 flex items-center space-x-4">
            <Link
              href="/cases/new"
              className="bg-brand-blue hover:bg-sky-600 text-white font-semibold px-5 py-2.5 rounded-xl shadow-md transition flex items-center space-x-2 text-sm"
            >
              <span>Create Diagnostic Case</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </div>

      {/* Pipeline Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* System Health */}
        <div className="bg-white p-6 rounded-xl border border-industrial-200 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-industrial-500">API Gateway</span>
            <Server className="w-4 h-4 text-industrial-400" />
          </div>
          <div className="flex items-center space-x-2">
            {health?.status === "healthy" ? (
              <>
                <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                <span className="font-semibold text-industrial-800">Operational</span>
              </>
            ) : (
              <>
                <AlertCircle className="w-5 h-5 text-amber-500" />
                <span className="font-semibold text-industrial-800">
                  {loading ? "Checking..." : "Offline / Mocking"}
                </span>
              </>
            )}
          </div>
          <p className="text-xs text-industrial-400">
            Environment: {health?.environment || "production"} (v{health?.version || "1.0.0"})
          </p>
        </div>

        {/* Multimodal Models */}
        <div className="bg-white p-6 rounded-xl border border-industrial-200 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-industrial-500">AI Perception</span>
            <Layers className="w-4 h-4 text-industrial-400" />
          </div>
          <div className="flex items-center space-x-2">
            {readiness?.ready ? (
              <>
                <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                <span className="font-semibold text-industrial-800">7 Subsystems Active</span>
              </>
            ) : (
              <>
                <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                <span className="font-semibold text-industrial-800">Vision, Audio, Sensors</span>
              </>
            )}
          </div>
          <p className="text-xs text-industrial-400">
            Cross-attention fusion & calibrated reasoning
          </p>
        </div>

        {/* RAG Knowledge Store */}
        <div className="bg-white p-6 rounded-xl border border-industrial-200 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-industrial-500">Knowledge RAG</span>
            <BookOpen className="w-4 h-4 text-industrial-400" />
          </div>
          <div className="flex items-center space-x-2">
            <CheckCircle2 className="w-5 h-5 text-emerald-500" />
            <span className="font-semibold text-industrial-800">OEM Manuals Indexed</span>
          </div>
          <p className="text-xs text-industrial-400">
            ISO 10816-3 & Technical SOP Vector Store
          </p>
        </div>
      </div>

      {/* Guided Workflow Steps */}
      <div className="bg-white rounded-xl border border-industrial-200 p-6 shadow-sm space-y-4">
        <h2 className="text-lg font-bold text-industrial-900">Multimodal Autonomous Diagnosis Workflow</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          <div className="p-4 bg-industrial-50 rounded-lg border border-industrial-100 space-y-1">
            <div className="text-brand-blue font-bold">Step 1: Ingest Evidence</div>
            <p className="text-industrial-600 text-xs">Submit equipment images, audio recordings, sensor telemetry, and notes.</p>
          </div>
          <div className="p-4 bg-industrial-50 rounded-lg border border-industrial-100 space-y-1">
            <div className="text-brand-blue font-bold">Step 2: Perception & Fusion</div>
            <p className="text-industrial-600 text-xs">Deep CNNs & MLPs extract embeddings and cross-attend across modalities.</p>
          </div>
          <div className="p-4 bg-industrial-50 rounded-lg border border-industrial-100 space-y-1">
            <div className="text-brand-blue font-bold">Step 3: RAG Retrieval</div>
            <p className="text-industrial-600 text-xs">Hybrid retrieval fetches relevant pages from maintenance manuals.</p>
          </div>
          <div className="p-4 bg-industrial-50 rounded-lg border border-industrial-100 space-y-1">
            <div className="text-brand-blue font-bold">Step 4: Auditable Report</div>
            <p className="text-industrial-600 text-xs">Synthesizes evidence citations, decomposed confidence, and actions.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
