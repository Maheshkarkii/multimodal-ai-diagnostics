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
  Server,
  Sparkles,
  Cpu,
  FileCheck2
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
    <div className="space-y-10">
      {/* Hero Section - Apple Style White Canvas */}
      <div className="relative overflow-hidden bg-white rounded-[28px] border border-black/[0.06] p-8 sm:p-12 md:p-16 shadow-[0_4px_24px_rgba(0,0,0,0.03)] text-center">
        <div className="max-w-3xl mx-auto space-y-6">
          <div className="inline-flex items-center space-x-2 bg-[#f5f5f7] text-[#1d1d1f] border border-black/[0.06] px-3.5 py-1 rounded-full text-xs font-medium tracking-wide">
            <Sparkles className="w-3.5 h-3.5 text-[#1d1d1f]" />
            <span>Multimodal Autonomous Diagnostics</span>
          </div>
          
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-semibold tracking-tight text-[#1d1d1f]">
            Intelligence for every industrial asset.
          </h1>
          
          <p className="text-[#6e6e73] text-base sm:text-lg max-w-2xl mx-auto font-normal leading-relaxed">
            Autonomous diagnostic reasoning across vision saliency heatmaps, acoustic harmonics, physical telemetry thresholds, and verified OEM technical manuals.
          </p>
          
          <div className="pt-2 flex flex-wrap items-center justify-center gap-3">
            <Link
              href="/cases/new"
              className="bg-[#1d1d1f] hover:bg-black text-white font-medium px-6 py-3 rounded-full shadow-[0_2px_8px_rgba(0,0,0,0.12)] hover:shadow-[0_4px_16px_rgba(0,0,0,0.18)] transition-all duration-200 flex items-center space-x-2 text-sm active:scale-98"
            >
              <span>Create Diagnostic Case</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
            <a
              href="http://127.0.0.1:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="bg-[#f5f5f7] hover:bg-[#e8e8ed] text-[#1d1d1f] font-medium px-6 py-3 rounded-full border border-black/[0.06] transition-all duration-200 text-sm"
            >
              API Reference
            </a>
          </div>
        </div>
      </div>

      {/* System Status - Clean Modular Cards */}
      <div className="space-y-4">
        <div className="flex items-center justify-between px-2">
          <h2 className="text-sm font-semibold tracking-tight text-[#1d1d1f]">System Telemetry & Architecture</h2>
          <span className="text-xs text-[#86868b]">Real-time operational status</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* API Gateway */}
          <div className="bg-white p-6 rounded-[22px] border border-black/[0.06] shadow-[0_2px_12px_rgba(0,0,0,0.02)] hover:shadow-[0_8px_24px_rgba(0,0,0,0.05)] transition-all duration-300 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-[#86868b]">Core Backend</span>
              <div className="w-8 h-8 rounded-full bg-[#f5f5f7] flex items-center justify-center text-[#1d1d1f]">
                <Server className="w-4 h-4" />
              </div>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                {health?.status === "healthy" ? (
                  <>
                    <div className="w-2 h-2 rounded-full bg-[#34c759] animate-pulse" />
                    <span className="text-lg font-semibold tracking-tight text-[#1d1d1f]">Operational</span>
                  </>
                ) : (
                  <>
                    <div className="w-2 h-2 rounded-full bg-[#f5a623]" />
                    <span className="text-lg font-semibold tracking-tight text-[#1d1d1f]">
                      {loading ? "Checking..." : "Offline / Mocking"}
                    </span>
                  </>
                )}
              </div>
              <p className="text-xs text-[#6e6e73] mt-1">
                FastAPI Gateway · {health?.environment || "production"} (v{health?.version || "1.0.0"})
              </p>
            </div>
          </div>

          {/* Multimodal Models */}
          <div className="bg-white p-6 rounded-[22px] border border-black/[0.06] shadow-[0_2px_12px_rgba(0,0,0,0.02)] hover:shadow-[0_8px_24px_rgba(0,0,0,0.05)] transition-all duration-300 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-[#86868b]">AI Perception</span>
              <div className="w-8 h-8 rounded-full bg-[#f5f5f7] flex items-center justify-center text-[#1d1d1f]">
                <Layers className="w-4 h-4" />
              </div>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-[#34c759]" />
                <span className="text-lg font-semibold tracking-tight text-[#1d1d1f]">
                  {readiness?.ready ? "7 Subsystems Active" : "Vision, Audio & Sensors"}
                </span>
              </div>
              <p className="text-xs text-[#6e6e73] mt-1">
                Cross-attention fusion & calibrated reasoning
              </p>
            </div>
          </div>

          {/* RAG Knowledge Store */}
          <div className="bg-white p-6 rounded-[22px] border border-black/[0.06] shadow-[0_2px_12px_rgba(0,0,0,0.02)] hover:shadow-[0_8px_24px_rgba(0,0,0,0.05)] transition-all duration-300 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-[#86868b]">Knowledge Index</span>
              <div className="w-8 h-8 rounded-full bg-[#f5f5f7] flex items-center justify-center text-[#1d1d1f]">
                <BookOpen className="w-4 h-4" />
              </div>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-[#34c759]" />
                <span className="text-lg font-semibold tracking-tight text-[#1d1d1f]">OEM Manuals Indexed</span>
              </div>
              <p className="text-xs text-[#6e6e73] mt-1">
                ISO 10816-3 & Technical SOP Dense Vector Store
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Guided Workflow Steps - Minimalist Apple Grid */}
      <div className="bg-white rounded-[28px] border border-black/[0.06] p-8 sm:p-10 shadow-[0_4px_20px_rgba(0,0,0,0.02)] space-y-6">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-[#1d1d1f]">
            Autonomous Diagnostic Pipeline
          </h2>
          <p className="text-xs text-[#6e6e73] mt-0.5">End-to-end multi-stage reasoning architecture</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-5 bg-[#f5f5f7] rounded-[18px] border border-black/[0.04] space-y-2 hover:bg-[#fafafc] transition-colors">
            <div className="w-7 h-7 rounded-full bg-white flex items-center justify-center text-xs font-semibold text-[#1d1d1f] shadow-xs">
              1
            </div>
            <div className="font-semibold text-sm text-[#1d1d1f]">Ingest Evidence</div>
            <p className="text-[#6e6e73] text-xs leading-relaxed">
              Capture inspection imagery, acoustic recordings, sensor telemetry, and technician logs.
            </p>
          </div>

          <div className="p-5 bg-[#f5f5f7] rounded-[18px] border border-black/[0.04] space-y-2 hover:bg-[#fafafc] transition-colors">
            <div className="w-7 h-7 rounded-full bg-white flex items-center justify-center text-xs font-semibold text-[#1d1d1f] shadow-xs">
              2
            </div>
            <div className="font-semibold text-sm text-[#1d1d1f]">Perception & Fusion</div>
            <p className="text-[#6e6e73] text-xs leading-relaxed">
              Deep CNNs, STFT spectrograms, and sensor MLPs produce aligned unified feature representations.
            </p>
          </div>

          <div className="p-5 bg-[#f5f5f7] rounded-[18px] border border-black/[0.04] space-y-2 hover:bg-[#fafafc] transition-colors">
            <div className="w-7 h-7 rounded-full bg-white flex items-center justify-center text-xs font-semibold text-[#1d1d1f] shadow-xs">
              3
            </div>
            <div className="font-semibold text-sm text-[#1d1d1f]">RAG Retrieval</div>
            <p className="text-[#6e6e73] text-xs leading-relaxed">
              Dense + BM25 hybrid search retrieves verified manufacturer repair procedures.
            </p>
          </div>

          <div className="p-5 bg-[#f5f5f7] rounded-[18px] border border-black/[0.04] space-y-2 hover:bg-[#fafafc] transition-colors">
            <div className="w-7 h-7 rounded-full bg-white flex items-center justify-center text-xs font-semibold text-[#1d1d1f] shadow-xs">
              4
            </div>
            <div className="font-semibold text-sm text-[#1d1d1f]">Auditable Report</div>
            <p className="text-[#6e6e73] text-xs leading-relaxed">
              Generates calibrated confidence scores, Grad-CAM heatmaps, and evidence audit trails.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
