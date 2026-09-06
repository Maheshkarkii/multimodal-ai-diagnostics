"use client";

import Link from "next/link";
import { Wrench, Activity, Plus } from "lucide-react";

export function Navbar() {
  return (
    <header className="sticky top-0 z-50 bg-[#f5f5f7]/80 backdrop-blur-xl border-b border-black/[0.06] transition-all">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Link href="/" className="flex items-center space-x-2.5 group">
            <div className="w-8 h-8 rounded-full bg-[#1d1d1f] text-white flex items-center justify-center shadow-sm group-hover:scale-105 transition-transform duration-200">
              <Wrench className="w-4 h-4 text-white" />
            </div>
            <div className="flex items-center space-x-2">
              <span className="font-semibold text-[15px] tracking-tight text-[#1d1d1f]">
                AI Field Engineer
              </span>
              <span className="text-[11px] font-medium bg-black/[0.04] text-[#6e6e73] px-2 py-0.5 rounded-full border border-black/[0.04]">
                1.0
              </span>
            </div>
          </Link>
        </div>

        <nav className="flex items-center space-x-4 text-[13px] font-normal">
          <Link
            href="/"
            className="text-[#6e6e73] hover:text-[#1d1d1f] transition-colors flex items-center space-x-1.5 px-3 py-1.5 rounded-full hover:bg-black/[0.03]"
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Dashboard</span>
          </Link>
          <Link
            href="/cases/new"
            className="bg-[#1d1d1f] hover:bg-black text-white px-4 py-1.5 rounded-full font-medium text-[13px] transition-all duration-200 flex items-center space-x-1.5 shadow-[0_1px_3px_rgba(0,0,0,0.12)] active:scale-95"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>New Case</span>
          </Link>
        </nav>
      </div>
    </header>
  );
}
